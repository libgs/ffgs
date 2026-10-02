"""A tiny model + processor for exercising the pipeline without a real model."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn

from ffgs import (
    Cameras,
    Gaussians,
    ImageFitConfig,
    ModelSpec,
    Prepared,
    Processor,
    Views,
    register_model,
)
from ffgs.geometry import ModelFrame, anchored_frame, quat_to_matrix
from ffgs.types import sh_degree_from_channels

MODEL_TYPE = "ffgs-test-dummy"
POSE_FREE_MODEL_TYPE = "ffgs-test-dummy-pose-free"


class DummyModel(nn.Module, PyTorchModelHubMixin):
    """Packed Gaussians [B, N, 11 + C] from pooled images and camera centres."""

    def __init__(self, num_gaussians: int = 5, sh_degree: int | None = 1) -> None:
        super().__init__()
        self.num_gaussians = num_gaussians
        self.sh_degree = sh_degree
        colors = 3 if sh_degree is None else 3 * (sh_degree + 1) ** 2
        self.proj = nn.Linear(6, num_gaussians * (11 + colors))

    def forward(self, model_input: dict[str, torch.Tensor]) -> torch.Tensor:
        pooled = model_input["images"].mean(dim=(1, 3, 4))  # [B, 3]
        centres = model_input["c2w"][:, :, :3, 3].mean(dim=1)  # [B, 3]
        out = self.proj(torch.cat((pooled, centres), dim=-1))
        return out.reshape(out.shape[0], self.num_gaussians, -1)


@dataclass
class DummyProcessorConfig(ImageFitConfig):
    image_shape: list[int] = field(default_factory=lambda: [16, 32])
    scene_scale: float = 0.2
    znear: float = 0.025
    zfar: float = 125.0


class DummyProcessor(Processor):
    config_cls = DummyProcessorConfig

    def preprocess(
        self, views: Views, device: torch.device, scene_scale: float | None = None
    ) -> Prepared:
        scale = self.config.scene_scale if scene_scale is None else scene_scale
        frame = anchored_frame(views.c2w, scale)
        fitted = self.fit_views(views)
        model_input = {
            "images": fitted.images.float().to(device),
            "c2w": frame.c2w_to_model(fitted.c2w.float()).to(device),
        }
        return Prepared(model_input, frame)

    def postprocess(self, model_output: torch.Tensor) -> Gaussians:
        return gaussians_from_packed(model_output)

    def render_planes(self, frame: ModelFrame) -> tuple[float, float]:
        return self.config.znear / frame.scale, self.config.zfar / frame.scale


def gaussians_from_packed(packed: torch.Tensor) -> Gaussians:
    b, n, channels = packed.shape
    sh_degree = sh_degree_from_channels(channels - 11)
    colors = packed[..., 11:]
    if sh_degree is not None:
        colors = colors.reshape(b, n, (sh_degree + 1) ** 2, 3)
    return Gaussians(
        means=packed[..., 0:3],
        opacities=torch.sigmoid(packed[..., 3]),
        scales=torch.exp(packed[..., 4:7]),
        quats=torch.nn.functional.normalize(packed[..., 7:11], dim=-1),
        colors=colors,
        sh_degree=sh_degree,
    )


SPEC = register_model(
    ModelSpec(model_type=MODEL_TYPE, model_cls=DummyModel, processor_cls=DummyProcessor)
)


class PoseFreeDummyModel(nn.Module, PyTorchModelHubMixin):
    """Images only in; packed Gaussians and per-view cameras (wxyz quat +
    translation) out, both in its own frame: cameras near the identity rotation,
    spread along x, the Gaussians a few units in front of them."""

    def __init__(self, num_gaussians: int = 6, sh_degree: int | None = 2) -> None:
        super().__init__()
        self.num_gaussians = num_gaussians
        self.sh_degree = sh_degree
        colors = 3 if sh_degree is None else 3 * (sh_degree + 1) ** 2
        self.gaussians = nn.Linear(3, num_gaussians * (11 + colors))
        self.cameras = nn.Linear(3, 7)

    def forward(self, model_input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        per_view = model_input["images"].mean(dim=(3, 4))  # [B, V, 3]
        packed = self.gaussians(per_view.mean(dim=1))
        packed = packed.reshape(packed.shape[0], self.num_gaussians, -1)
        packed = torch.cat(
            [packed[..., :2], packed[..., 2:3] + 4.0, packed[..., 3:]], -1
        )
        cameras = self.cameras(per_view)
        quat = torch.cat(
            [torch.ones_like(cameras[..., :1]), 0.1 * cameras[..., :3]], -1
        )
        offset = torch.arange(per_view.shape[1], dtype=packed.dtype)[:, None]
        offset = offset.to(packed.device) * packed.new_tensor([0.5, 0.0, 0.0])
        translation = 0.3 * cameras[..., 3:6] + offset
        return {"gaussians": packed, "cameras": torch.cat([quat, translation], -1)}


@dataclass
class PoseFreeDummyProcessorConfig(ImageFitConfig):
    image_shape: list[int] = field(default_factory=lambda: [16, 32])
    znear: float = 0.01
    zfar: float = 100.0


class PoseFreeDummyProcessor(Processor):
    config_cls = PoseFreeDummyProcessorConfig

    def preprocess(self, views: Views, device: torch.device) -> Prepared:
        fitted = self.fit_views(views)
        return Prepared({"images": fitted.images.float().to(device)}, None)

    def postprocess(self, model_output: dict[str, torch.Tensor]) -> Gaussians:
        return gaussians_from_packed(model_output["gaussians"])

    def predicted_cameras(self, model_output: dict[str, torch.Tensor]) -> Cameras:
        cameras = model_output["cameras"]
        c2w = torch.eye(4).repeat(*cameras.shape[:2], 1, 1)
        c2w[..., :3, :3] = quat_to_matrix(cameras[..., :4])
        c2w[..., :3, 3] = cameras[..., 4:]
        k = torch.tensor([[1.2, 0, 0.5], [0, 1.2, 0.5], [0, 0, 1]])
        k = k.expand(*cameras.shape[:2], 3, 3)
        return Cameras(c2w, k, tuple(self.config.image_shape), True)

    def render_planes(self, frame: ModelFrame | None) -> tuple:
        if frame is None:
            return self.config.znear, self.config.zfar
        return self.config.znear / frame.scale, self.config.zfar / frame.scale


POSE_FREE_SPEC = register_model(
    ModelSpec(
        model_type=POSE_FREE_MODEL_TYPE,
        model_cls=PoseFreeDummyModel,
        processor_cls=PoseFreeDummyProcessor,
    )
)
