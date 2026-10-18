"""C3G preprocessing and nominal first-camera/unit-endpoint-baseline frame.

Matches the official 224-square path: PIL LANCZOS cover resize (round), centered
crop, then (x - 0.5) / 0.5. The network applies a second normalization itself.
Poses never condition the network; they only define nominal world placement.
Without poses both frame and predicted cameras are absent. This is not an
estimate of metric scale, nor the official evaluation-only pose optimization.
Render planes are nominal RE10K world units (model units without poses).
gsplat uses its default eps2d=0.3; its clipping and numerical rasterization do
not exactly reproduce the official scale-invariant CUDA renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from ...geometry import ModelFrame
from ...processor import ImageFitConfig, Prepared, Processor
from ...types import Gaussians, Views
from .modeling import C3GOutput


@dataclass
class C3GProcessorConfig(ImageFitConfig):
    image_shape: list[int] = field(default_factory=lambda: [224, 224])
    resize_mode: str = "lanczos"
    znear: float = 0.1
    zfar: float = 100.0
    background: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    radius_clip: float = 0.0
    rasterize_mode: str = "classic"
    clamp: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(size <= 0 or size % 14 for size in self.image_shape):
            raise ValueError("image_shape must contain positive multiples of 14")
        if not 0 < self.znear < self.zfar:
            raise ValueError("require 0 < znear < zfar")
        if self.rasterize_mode not in ("classic", "antialiased"):
            raise ValueError("rasterize_mode must be classic or antialiased")


class C3GProcessor(Processor):
    config_cls = C3GProcessorConfig
    camera_input = "none"

    def preprocess(self, views: Views, device: torch.device) -> Prepared:
        if views.images.shape[1] < 2:
            raise ValueError("C3G multiview inference needs at least 2 views")
        frame = None
        if views.c2w is not None:
            poses = views.c2w.to(device)
            baseline = torch.linalg.vector_norm(
                poses[:, -1, :3, 3] - poses[:, 0, :3, 3], dim=-1
            )
            if (
                not torch.isfinite(poses).all()
                or not ((baseline >= 1e-3) & (baseline <= 1e10)).all()
            ):
                raise ValueError(
                    "C3G requires finite poses and endpoint baseline in [1e-3, 1e10]"
                )
            frame = ModelFrame(poses[:, 0].clone(), baseline.reciprocal())
        images = self.fit_views(views).images
        return Prepared(((images - 0.5) / 0.5).to(device), frame)

    def postprocess(self, model_output: C3GOutput) -> Gaussians:
        g = model_output
        return Gaussians(
            means=g.means.float(),
            scales=g.scales.float(),
            quats=g.rotations[..., [3, 0, 1, 2]].float(),  # xyzw -> wxyz
            opacities=g.opacities.float(),
            colors=g.harmonics.transpose(-1, -2).float(),  # [B, N, 1, 3]
            sh_degree=0,
        )

    def render_planes(self, frame: ModelFrame | None) -> tuple[float, float]:
        return self.config.znear, self.config.zfar

    @property
    def background(self) -> tuple[float, float, float]:
        return tuple(self.config.background)

    @property
    def render_settings(self) -> dict[str, Any]:
        return {
            "radius_clip": self.config.radius_clip,
            "rasterize_mode": self.config.rasterize_mode,
            "clamp": self.config.clamp,
        }
