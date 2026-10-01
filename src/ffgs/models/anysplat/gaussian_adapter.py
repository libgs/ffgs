# Copyright (c) 2025 Lihan Jiang and Yucheng Mao
# SPDX-License-Identifier: MIT (see the AnySplat section of LICENSE, this directory)
#
# Modified by the libgs contributors, 2026, from
# src/model/encoder/common/gaussian_adapter.py and src/model/types.py of
# https://github.com/InternRobotics/AnySplat (5f5e208): inference-only subset. Only
# `UnifiedGaussianAdapter` (the adapter of `pose_free=True`) and what it uses from
# its base class are kept; the covariances (`build_covariance`) are not built, as
# ffgs renders from scales and quaternions; the unused `intrinsics` / `coordinates`
# arguments of `forward` dropped; einops' rearrange replaced by the equivalent
# `unflatten`, jaxtyping annotations by plain `Tensor`.

"""Raw Gaussian features -> Gaussian parameters (AnySplat's pose-free adapter)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float
    sh_degree: int


@dataclass
class Gaussians:
    """Upstream's Gaussians, without covariances. Quaternions are xyzw (scipy
    order); harmonics are [batch, gaussian, 3, d_sh]."""

    means: Tensor  # [batch, gaussian, 3]
    harmonics: Tensor  # [batch, gaussian, 3, d_sh]
    opacities: Tensor  # [batch, gaussian]
    scales: Tensor  # [batch, gaussian, 3]
    rotations: Tensor  # [batch, gaussian, 4], xyzw


class GaussianAdapter(nn.Module):
    cfg: GaussianAdapterCfg

    def __init__(self, cfg: GaussianAdapterCfg):
        super().__init__()
        self.cfg = cfg

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    @property
    def d_sh(self) -> int:
        return (self.cfg.sh_degree + 1) ** 2

    @property
    def d_in(self) -> int:
        return 7 + 3 * self.d_sh


class UnifiedGaussianAdapter(GaussianAdapter):
    def forward(
        self,
        means: Tensor,
        depths: Tensor,
        opacities: Tensor,
        raw_gaussians: Tensor,
        eps: float = 1e-8,
    ) -> Gaussians:
        scales, rotations, sh = raw_gaussians.split((3, 4, 3 * self.d_sh), dim=-1)

        scales = 0.001 * F.softplus(scales)
        scales = scales.clamp_max(0.3)

        # Normalize the quaternion features to yield a valid quaternion.
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)

        sh = sh.unflatten(-1, (3, self.d_sh))
        sh = sh.broadcast_to((*opacities.shape, 3, self.d_sh)) * self.sh_mask

        return Gaussians(
            means=means.float(),
            harmonics=sh.float(),
            opacities=opacities.float(),
            scales=scales.float(),
            rotations=rotations.float(),
        )
