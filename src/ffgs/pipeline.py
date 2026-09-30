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

from . import render as render_mod
from .geometry import ModelFrame
from .processor import Processor, read_json
from .registry import get_model_spec, spec_for_model
from .types import Cameras, Gaussians, Views

CONFIG_NAME = "config.json"


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
        **hub_kwargs: Any,
    ) -> GSPipeline:
        """`pretrained`: a directory written by `save_pretrained`, or a Hub repo id.

        `model_type` overrides the one in config.json (for directories exported
        before it was recorded).
        """
        if model_type is None:
            model_type = read_json(pretrained, CONFIG_NAME, **hub_kwargs).get(
                "model_type"
            )
            if model_type is None:
                raise ValueError(
                    f"{CONFIG_NAME} of {pretrained} has no model_type; pass "
                    "model_type= or re-export with GSPipeline.save_pretrained"
                )
        spec = get_model_spec(model_type)
        model = spec.model_cls.from_pretrained(str(pretrained), **hub_kwargs)
        processor = spec.processor_cls.from_pretrained(pretrained, **hub_kwargs)
        if processor_overrides:
            processor = spec.processor_cls(processor.config, **processor_overrides)
        pipe = cls(model, processor, model_type)
        return pipe.to(device) if device is not None else pipe

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> Path:
        """Weights + config.json (with model_type) + processor_config.json."""
        save_directory = Path(save_directory)
        self.model.save_pretrained(save_directory, **kwargs)
        config_path = save_directory / CONFIG_NAME
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        config["model_type"] = self.model_type
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
