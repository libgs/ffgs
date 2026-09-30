"""`GSPipeline`: posed images in, world-frame Gaussians out, for any registered model."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import dynamic
from . import render as render_mod
from .geometry import ModelFrame
from .processor import Processor, read_json
from .registry import ModelSpec, get_model_spec, spec_for_model
from .types import Cameras, Gaussians, Views

CONFIG_NAME = "config.json"
# Options of `huggingface_hub` downloads that `from_pretrained` passes through.
HUB_KWARGS = frozenset(
    {"revision", "cache_dir", "token", "force_download", "local_files_only"}
)


@dataclass
class GSOutput:
    gaussians: Gaussians  # world frame, fp32
    frame: ModelFrame
    # The model's own output in its working frame, when asked for.
    model_gaussians: Gaussians | None = None


class GSPipeline:
    """Load with `from_pretrained`, call on `Views`, render at `Cameras`.

    >>> pipe = GSPipeline.from_pretrained("path/or/repo-id", device="cuda")
    >>> out = pipe(Views(images, intrinsics, c2w))
    >>> images = pipe.render(out, Cameras(c2w_new, intrinsics_new, (h, w)))["images"]
    >>> out.gaussians.save_ply("scene.ply")
    """

    def __init__(
        self, model: nn.Module, processor: Processor, model_type: str | None = None
    ) -> None:
        self.model = model.eval()
        self.processor = processor
        self.model_type = model_type or spec_for_model(model).model_type

    # --- loading / saving -------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained: str | Path,
        *,
        model_type: str | None = None,
        device: torch.device | str | None = None,
        processor_overrides: dict[str, Any] | None = None,
        trust_remote_code: bool = False,
        **hub_kwargs: Any,
    ) -> GSPipeline:
        """`pretrained`: a directory written by `save_pretrained`, or a Hub repo id.

        `model_type` overrides the one in config.json (for directories exported
        before it was recorded). It resolves to a registered or built-in model,
        then an installed `ffgs.models` plugin, then — only with
        `trust_remote_code=True` — the code the repo lists under
        config.json:auto_map (see `ffgs.registry`). Pin `revision=` when trusting
        Hub code.

        `hub_kwargs` (`revision`, `cache_dir`, `token`, `force_download`,
        `local_files_only`) apply to every file fetched from the Hub: config,
        weights, processor config and code.
        """
        unknown = set(hub_kwargs) - HUB_KWARGS
        if unknown:
            raise TypeError(
                f"unexpected arguments {sorted(unknown)}; Hub options are "
                f"{sorted(HUB_KWARGS)}"
            )
        config = None
        if model_type is None:
            config = read_json(pretrained, CONFIG_NAME, **hub_kwargs)
            model_type = config.get("model_type")
            if model_type is None:
                raise ValueError(
                    f"{CONFIG_NAME} of {pretrained} has no model_type; pass "
                    "model_type= or re-export with GSPipeline.save_pretrained"
                )
        spec = _resolve_spec(
            model_type, pretrained, config, trust_remote_code, **hub_kwargs
        )
        model = spec.model_cls.from_pretrained(str(pretrained), **hub_kwargs)
        processor = spec.processor_cls.from_pretrained(pretrained, **hub_kwargs)
        if processor_overrides:
            processor = spec.processor_cls(processor.config, **processor_overrides)
        pipe = cls(model, processor, model_type)
        return pipe.to(device) if device is not None else pipe

    def save_pretrained(
        self, save_directory: str | Path, *, include_code: bool = False, **kwargs: Any
    ) -> Path:
        """Weights + config.json (with model_type) + processor_config.json; for a
        model loaded from repo code, that code too (+ config.json:auto_map).

        `include_code=True` also ships the code of an installed model: the package
        holding the model and processor classes is copied in and listed under
        config.json:auto_map, so the directory loads with `trust_remote_code=True`
        where that package is not installed (see `ffgs.dynamic.export_code`)."""
        save_directory = Path(save_directory)
        model_cls, processor_cls = type(self.model), type(self.processor)
        # Code first: a package that cannot be shipped fails before any write.
        auto_map = dynamic.save_code(model_cls, processor_cls, save_directory)
        if auto_map is None and include_code:
            auto_map = dynamic.export_code(model_cls, processor_cls, save_directory)
        self.model.save_pretrained(save_directory, **kwargs)
        config_path = save_directory / CONFIG_NAME
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        config["model_type"] = self.model_type
        if auto_map is not None:
            config["auto_map"] = auto_map
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        self.processor.save_pretrained(save_directory)
        return save_directory

    def to(self, device: torch.device | str) -> GSPipeline:
        self.model.to(device)
        return self

    @property
    def device(self) -> torch.device:
        """Where the model is: its first parameter or buffer, else the CPU."""
        tensor = next(chain(self.model.parameters(), self.model.buffers()), None)
        return torch.device("cpu") if tensor is None else tensor.device

    # --- inference --------------------------------------------------------------

    def autocast(self):
        dtype = self.processor.autocast_dtype
        if dtype is None or self.device.type != "cuda":
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        views: Views,
        *,
        return_model_frame: bool = False,
        **options: Any,
    ) -> GSOutput:
        """Predict Gaussians from `views`. `options` go to the processor
        (e.g. `scene_scale=`)."""
        with torch.autocast(device_type=self.device.type, enabled=False):
            prepared = self.processor.preprocess(views, self.device, **options)
        with self.autocast():
            raw = self.model(prepared.model_input)
        model_gaussians = self.processor.postprocess(raw)
        return GSOutput(
            gaussians=prepared.frame.gaussians_to_world(model_gaussians),
            frame=prepared.frame,
            model_gaussians=model_gaussians if return_model_frame else None,
        )

    @torch.no_grad()
    def render(
        self,
        output: GSOutput,
        cameras: Cameras,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Render world-frame Gaussians at world-frame `cameras`."""
        near, far = self.processor.render_planes(output.frame)
        kwargs.setdefault("background", self.processor.background)
        with torch.autocast(device_type=self.device.type, enabled=False):
            return render_mod.render(
                output.gaussians, cameras.to(self.device), near=near, far=far, **kwargs
            )


def _resolve_spec(
    model_type: str,
    pretrained: str | Path,
    config: dict | None,
    trust_remote_code: bool,
    **hub_kwargs: Any,
) -> ModelSpec:
    """Installed model first; repo code (config.json:auto_map) only if trusted."""
    try:
        return get_model_spec(model_type)
    except KeyError:
        if config is None:
            config = read_json(pretrained, CONFIG_NAME, **hub_kwargs)
        auto_map = config.get("auto_map")
        if auto_map is None:
            raise
    if not trust_remote_code:
        raise ValueError(
            f"model_type {model_type!r} is not installed, and {pretrained} ships "
            f"its own code for it ({CONFIG_NAME}:auto_map = {auto_map}). That code "
            "would run on this machine: read it, then pass trust_remote_code=True "
            "(with a pinned revision= for Hub repos). Or install a package that "
            f"provides {model_type!r}."
        )
    return dynamic.spec_from_auto_map(model_type, auto_map, pretrained, **hub_kwargs)
