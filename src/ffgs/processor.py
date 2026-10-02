"""Processor base: a model's pre / post-processing, stored as `processor_config.json`.

A processor owns everything between the standard `Views` / `Gaussians` and one
model's tensors: resizing to the training rule, normalising cameras into the
model's frame (the `ModelFrame` that also takes predictions back to the caller's
world frame; fitted to the given poses for models that predict cameras), and
building the model input. Its settings are plain JSON so they
travel with the weights.

Configs that subclass `ImageFitConfig` get `fit_views` / `fit_cameras`: resize +
centre crop by default (the training rule), or pad / pass-through, and optionally a
custom resize callable (`Processor(..., resize=fn)`, not serialised).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar

import torch

from .geometry import ModelFrame
from .image import (
    CROP_MODES,
    RESIZE_MODES,
    Resize,
    fit_images,
    fit_intrinsics,
)
from .types import Cameras, Gaussians, Views

PROCESSOR_CONFIG_NAME = "processor_config.json"


@dataclass
class Prepared:
    """A processor's output for one call: what the model eats, and the frame.

    `frame` is None for models that predict their cameras: it is then fitted
    after the model, from the predicted cameras (see `Processor.output_frame`).
    """

    model_input: Any
    frame: ModelFrame | None


@dataclass
class ImageFitConfig:
    """How inputs are brought to the model's input shape. Subclass per model and
    re-declare fields to change their defaults."""

    # [h, w] the model sees.
    image_shape: list[int] = field(default_factory=lambda: [256, 256])
    # Built-in resize kernel (see `ffgs.image.rescale`).
    resize_mode: str = "bilinear"
    # "crop": cover + centre crop (the training rule); "pad": fit inside + pad;
    # "none": inputs are already `image_shape`.
    crop_mode: str = "crop"
    # Fill value of the padded border for crop_mode="pad".
    pad_value: float = 0.0

    def __post_init__(self) -> None:
        self.image_shape = [int(x) for x in self.image_shape]
        if len(self.image_shape) != 2:
            raise ValueError(f"image_shape must be [h, w], got {self.image_shape}")
        if self.resize_mode not in RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {RESIZE_MODES}")
        if self.crop_mode not in CROP_MODES:
            raise ValueError(f"crop_mode must be one of {CROP_MODES}")


class Processor:
    config_cls: ClassVar[type]

    def __init__(
        self,
        config: Any | None = None,
        *,
        resize: Resize | None = None,
        **overrides: Any,
    ) -> None:
        if config is None:
            config = self.config_cls()
        elif isinstance(config, dict):
            config = self.config_cls(**config)
        if overrides:
            config = self.config_cls(**{**asdict(config), **overrides})
        self.config = config
        # A custom resize callable for `fit_views`; replaces `resize_mode`.
        self.resize = resize

    # --- fitting inputs (configs subclassing ImageFitConfig) ----------------------

    def fit_views(self, views: Views, image_shape: list[int] | None = None) -> Views:
        """Bring `views` to `image_shape` (default: the model's) by the config's
        `crop_mode`. Also the way to derive ground truth for target views. Returns
        normalised intrinsics (None if `views` had none); c2w untouched."""
        cfg = self._fit_config()
        shape = tuple(image_shape or cfg.image_shape)
        if views.intrinsics is None:
            # The images are fitted the same way whatever K is; any valid one does.
            source_k = torch.tensor(
                [[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]], device=views.images.device
            ).expand(*views.images.shape[:2], 3, 3)
        else:
            source_k = views.normalized_k
        images, intrinsics = [], []
        for i in range(views.images.shape[0]):
            img, k = fit_images(
                views.images[i],
                source_k[i],
                shape,
                resize_mode=cfg.resize_mode,
                crop_mode=cfg.crop_mode,
                resize=self.resize,
                pad_value=cfg.pad_value,
            )
            images.append(img)
            intrinsics.append(k)
        return Views(
            torch.stack(images),
            None if views.intrinsics is None else torch.stack(intrinsics),
            views.c2w,
            normalized_intrinsics=True,
        )

    def fit_cameras(
        self, cameras: Cameras, image_shape: list[int] | None = None
    ) -> Cameras:
        """`fit_views` for cameras alone: where to render to compare against
        `fit_views` images. `cameras.image_shape` is the source image size."""
        cfg = self._fit_config()
        shape = tuple(image_shape or cfg.image_shape)
        k = fit_intrinsics(
            cameras.normalized_k, cameras.image_shape, shape, cfg.crop_mode
        )
        return Cameras(cameras.c2w, k, shape, normalized_intrinsics=True)

    def _fit_config(self) -> ImageFitConfig:
        if not isinstance(self.config, ImageFitConfig):
            raise TypeError(
                f"{type(self.config).__name__} does not subclass ImageFitConfig"
            )
        return self.config

    # --- serialisation ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def save_pretrained(self, save_directory: str | Path) -> Path:
        path = Path(save_directory) / PROCESSOR_CONFIG_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def from_pretrained(cls, pretrained: str | Path, **hub_kwargs: Any) -> Processor:
        data = read_json(pretrained, PROCESSOR_CONFIG_NAME, **hub_kwargs)
        return cls.from_dict(data, source=PROCESSOR_CONFIG_NAME)

    @classmethod
    def from_dict(cls, data: dict[str, Any], source: str = "config") -> Processor:
        known = {f.name for f in fields(cls.config_cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"{source} has keys {sorted(unknown)} unknown to {cls.__name__}"
            )
        return cls(data)

    # --- the per-model contract -------------------------------------------------

    def preprocess(
        self, views: Views, device: torch.device, **options: Any
    ) -> Prepared:
        raise NotImplementedError

    def postprocess(self, model_output: Any) -> Gaussians:
        """Raw model output -> model-frame `Gaussians`."""
        raise NotImplementedError

    def predicted_cameras(self, model_output: Any) -> Cameras | None:
        """The input cameras a model predicts, in its frame (normalised K for the
        model's input images); None for models that take cameras as input."""
        return None

    def output_frame(
        self, views: Views, prepared: Prepared, cameras: Cameras | None
    ) -> ModelFrame | None:
        """The model frame of this call. Posed models fix it in `preprocess`;
        for models that predict cameras it is fitted to the given poses over all
        views (`ModelFrame.fit`), and None when there are none."""
        if prepared.frame is not None:
            return prepared.frame
        if cameras is None or views.c2w is None:
            return None
        return ModelFrame.fit(cameras.c2w, views.c2w)

    def render_planes(self, frame: ModelFrame | None) -> tuple[float, float]:
        """(near, far) for Gaussians predicted in `frame`: in world units, or in
        model units when there is no frame (no input poses)."""
        raise NotImplementedError

    @property
    def background(self) -> tuple[float, float, float] | None:
        return None

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        return None


def read_json(pretrained: str | Path, filename: str, **hub_kwargs: Any) -> dict:
    """`filename` from a local directory or an HF Hub repo id."""
    local = Path(pretrained)
    if local.is_dir():
        path = local / filename
        if not path.is_file():
            raise FileNotFoundError(f"{filename} not found in {local}")
    else:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(repo_id=str(pretrained), filename=filename, **hub_kwargs)
        )
    return json.loads(path.read_text())
