"""Standard inference for feed-forward 3D Gaussian Splatting models.

Posed images in (`Views`), Gaussians out (in the model's frame, or the caller's
world frame with `out.to_world()`), one `GSPipeline.from_pretrained` for every
registered model. See README.md.

Depends on torch, torchvision, numpy, pillow, huggingface_hub and safetensors;
gsplat is imported only when rendering (`pip install ffgs[render]`).
"""

import importlib.metadata as _metadata

from .geometry import ModelFrame
from .pipeline import GSOutput, GSPipeline
from .processor import ImageFitConfig, Prepared, Processor
from .registry import ModelSpec, get_model_spec, register_model
from .render import render
from .types import Cameras, Gaussians, Views

try:
    __version__ = _metadata.version("ffgs")
except _metadata.PackageNotFoundError:  # a source tree that is not installed
    __version__ = "0+unknown"

__all__ = [
    "Cameras",
    "GSOutput",
    "GSPipeline",
    "Gaussians",
    "ImageFitConfig",
    "ModelFrame",
    "ModelSpec",
    "Prepared",
    "Processor",
    "Views",
    "get_model_spec",
    "register_model",
    "render",
]
