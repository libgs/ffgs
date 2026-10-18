"""Gaussian-only C3G, adapted from encoder_vggt.py @ 40b2cc9 (MIT).

Modified for ffgs: standalone inference module, configurable tiny architecture
for CPU references, no training/feature branches, no visualization DPT forward.
All released checkpoint parameters, including DPT, retain their upstream names.
Inputs have already undergone C3G's (x - 0.5) / 0.5 normalization; Aggregator
then applies ImageNet normalization again, exactly as the official inference path.
See LICENSE for source and non-commercial weight restrictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub import PyTorchModelHubMixin
from torch import nn
from torch.nn import functional as F

from .gmae import Transformer
from .vggt.aggregator import Aggregator
from .vggt.heads.dpt_head import DPTHead


@dataclass
class C3GOutput:
    """Upstream's Gaussian adapter output, per sample [B, N, ...]: rotations
    xyzw, harmonics [B, N, 3, 1] (SH degree 0)."""

    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    harmonics: torch.Tensor


class C3G(nn.Module, PyTorchModelHubMixin):
    """Multiview C3G: normalized images [B,V,3,H,W] -> model-frame Gaussians
    (upstream's adapter output, `C3GOutput`).

    Defaults match the released multiview checkpoint (1300 keys). The frozen
    DPT module is checkpoint-only; its visualization depth never feeds Gaussians.
    No pretrained foundation weights are downloaded during construction.
    """

    def __init__(
        self,
        num_gaussians: int = 2048,
        decoder_depth: int = 2,
        decoder_heads: int = 16,
        embed_dim: int = 1024,
        aggregator: dict[str, Any] | None = None,
        dpt_head: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        dimension = embed_dim * 2
        if num_gaussians <= 0 or dimension % decoder_heads:
            raise ValueError(
                "positive num_gaussians and divisible decoder_heads required"
            )
        self.backbone = nn.Module()
        self.backbone.aggregator = Aggregator(
            embed_dim=embed_dim, **dict(aggregator or {})
        )
        self.dpt_head = DPTHead(dimension, **dict(dpt_head or {}))
        self.dpt_head.requires_grad_(False)
        self.gaussian_tokens = nn.Parameter(torch.randn(num_gaussians, dimension))
        self.anchor_positions = nn.Parameter(
            torch.tensor([[0, 0, 1]]).repeat(num_gaussians, 1), requires_grad=False
        )
        self.gmae_decoder = Transformer(
            dim=dimension,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=dimension // decoder_heads,
            mlp_dim=dimension * 2,
        )
        self.gmae_to_gaussians = nn.Linear(dimension, 14)

    @property
    def sh_degree(self) -> int:
        return 0

    def forward(self, images: torch.Tensor) -> C3GOutput:
        tokens, patch_start = self.backbone.aggregator(images)
        features = tokens[-1][:, :, patch_start:].flatten(1, 2)
        queries = self.gaussian_tokens.unsqueeze(0).expand(images.shape[0], -1, -1)
        decoded = self.gmae_decoder(torch.cat((features, queries), dim=1))
        parameters = self.gmae_to_gaussians(decoded[:, -queries.shape[1] :])
        means = parameters[..., :3] + self.anchor_positions.unsqueeze(0)
        densities = parameters[..., 3].sigmoid()
        opacities = 0.5 * (1 - (1 - densities) ** 1.0 + densities**1.0)
        scales = 0.001 * F.softplus(parameters[..., 4:7])
        rotations = parameters[..., 7:11]
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
        return C3GOutput(
            means=means,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            harmonics=parameters[..., 11:14].unsqueeze(-1),
        )
