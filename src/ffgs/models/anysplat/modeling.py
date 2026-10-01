"""The AnySplat network: images -> Gaussians and the input cameras, in its frame."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .encoder import (
    EncoderAnySplat,
    EncoderAnySplatCfg,
    EncoderOutput,
    OpacityMappingCfg,
)
from .gaussian_adapter import GaussianAdapterCfg


class AnySplat(nn.Module, PyTorchModelHubMixin):
    """
    AnySplat (Jiang et al., TOG 2025): VGGT-based feed-forward Gaussians from
    unposed images, with the cameras it predicts for them.

    Input (from `AnySplatProcessor`): images [B, V, 3, H, W] in [0, 1], H and W
    multiples of the patch size. Output: upstream's `EncoderOutput` -- Gaussians
    (xyzw quaternions, SH of `sh_degree`) and `pred_context_pose` (c2w, normalised
    K) in the frame of the first predicted camera, at the model's own scale.

    The arguments are the inference fields of upstream's `encoder_cfg` (defaults:
    the released `lhjiang/anysplat` config) and the VGGT architecture (defaults:
    VGGT-1B). The aggregator is kept in bfloat16, as upstream; the parameter names
    are upstream's (`encoder.`...), so its checkpoint loads as it is.
    """

    def __init__(
        self,
        voxel_size: float = 0.002,
        voxelize: bool = True,
        sh_degree: int = 4,
        gaussian_scale_min: float = 0.5,
        gaussian_scale_max: float = 15.0,
        opacity_mapping: dict[str, float] | None = None,
        intermediate_layer_idx: list[int] | None = (4, 11, 17, 23),
        gs_prune: bool = False,
        opacity_threshold: float = 0.001,
        gs_keep_ratio: float = 1.0,
        render_conf: bool = False,
        opacity_conf: bool = False,
        conf_threshold: float = 0.1,
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        aggregator: dict[str, Any] | None = None,
        camera_head: dict[str, Any] | None = None,
        depth_head: dict[str, Any] | None = None,
        gs_head: dict[str, Any] | None = None,
    ):
        super().__init__()
        mapping = {"initial": 0.0, "final": 0.0, "warm_up": 1}
        cfg = EncoderAnySplatCfg(
            voxel_size=voxel_size,
            voxelize=voxelize,
            gaussian_adapter=GaussianAdapterCfg(
                gaussian_scale_min=gaussian_scale_min,
                gaussian_scale_max=gaussian_scale_max,
                sh_degree=sh_degree,
            ),
            opacity_mapping=OpacityMappingCfg(**(opacity_mapping or mapping)),
            intermediate_layer_idx=(
                None if intermediate_layer_idx is None else list(intermediate_layer_idx)
            ),
            gs_prune=gs_prune,
            opacity_threshold=opacity_threshold,
            gs_keep_ratio=gs_keep_ratio,
            render_conf=render_conf,
            opacity_conf=opacity_conf,
            conf_threshold=conf_threshold,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            aggregator=dict(aggregator or {}),
            camera_head=dict(camera_head or {}),
            depth_head=dict(depth_head or {}),
            gs_head=dict(gs_head or {}),
        )
        self.encoder = EncoderAnySplat(cfg)

    @property
    def sh_degree(self) -> int:
        return self.encoder.cfg.gaussian_adapter.sh_degree

    def forward(self, images: torch.Tensor) -> EncoderOutput:
        # Upstream's `AnySplat.inference`: the encoder at global step 0.
        return self.encoder(images, global_step=0)
