# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by the libgs contributors, 2026, from tokengs/models/activations.py of
# https://github.com/nv-tlabs/TokenGS (b16269c): the clip head only, taking a
# `TokenGSConfig`; einops replaced by the equivalent reshape / permute; the unused
# ray arguments and weight-decay flags removed. Parameter names are unchanged.

"""Gaussian tokens -> activated Gaussian parameters."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_

from .enc_dec import TokenGSConfig


class ClipActivationHead(nn.Module):
    """
    Clip-based activation head that uses exponential activations with hard
    clipping. Uses a single deconv (linear) layer for all Gaussian components.
    """

    def __init__(self, opt: TokenGSConfig):
        super().__init__()
        self.opt = opt

        self.num_gaussians_per_token = self.opt.dec_patch_size**2

        scale_cap = self.opt.gaussian_scale_cap
        self.scale_shift = 1 - math.log(scale_cap)
        self.scale_cap = scale_cap

        self.output_dims = 3 + 1 + 3 + 4 + 3  # x, y, z, opacity, scale, rotation, rgb

        # Single linear layer (deconv) for all components
        self.deconv = nn.Linear(
            self.opt.enc_embed_dim,
            self.output_dims * self.opt.dec_patch_size * self.opt.dec_patch_size,
            bias=True,
        )

        self._init_weights(self.deconv)
        if self.opt.clip_head_z_init is not None:
            bias_val = math.log(1.0 + self.opt.clip_head_z_init)
            with torch.no_grad():
                self.deconv.bias.view(
                    self.opt.dec_patch_size * self.opt.dec_patch_size,
                    self.output_dims,
                )[:, 2].fill_(bias_val)

    def _init_weights(self, m):
        """Initialize deconv weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.opt.clip_head_readout_std)
            if m.bias is not None:
                m.bias.data.zero_()

    def pos_act(self, x: torch.Tensor) -> torch.Tensor:
        """Position activation without scaling factor (for xyz prediction)"""
        pos = torch.sign(x) * (torch.expm1(torch.abs(x)))  # inverse log transform
        return pos

    def scale_act(self, x: torch.Tensor) -> torch.Tensor:
        """Scale activation with exponential and hard clipping"""
        x = x - self.scale_shift
        return torch.minimum(
            torch.exp(x),
            torch.tensor(self.scale_cap, device=x.device, dtype=x.dtype),
        )

    def opacity_act(self, x: torch.Tensor) -> torch.Tensor:
        """Opacity activation with hard clipping"""
        return torch.sigmoid(x - self.opt.opacity_bias)

    def rot_act(self, x: torch.Tensor) -> torch.Tensor:
        """Rotation normalization"""
        return F.normalize(x, dim=-1)

    def rgb_act(self, x: torch.Tensor) -> torch.Tensor:
        """RGB activation (always tanh for clip-based head)"""
        return 0.5 * torch.tanh(x) + 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert tokens to Gaussians using a single deconv layer.

        Args:
            x: Input tokens [B, N, C]

        Returns:
            Gaussians [B, N * num_gaussians_per_token, 14]: xyz, opacity, scale,
            rotation (wxyz), rgb
        """
        B = x.shape[0]

        x = self.deconv(x)  # [B, N, output_dims * P * P]
        # "b n (p c) -> b c (n p)"
        n = x.shape[1]
        p = self.opt.dec_patch_size**2
        x = x.reshape(B, n, p, -1).permute(0, 3, 1, 2).reshape(B, -1, n * p)

        x = x.reshape(B, self.output_dims, -1)  # B, output_dims, N * P * P
        x = x.permute(0, 2, 1).contiguous()  # B, N * P * P, output_dims

        pos, rgb, scaling, rotation, opacity = x.split([3, 3, 3, 4, 1], dim=-1)
        pos = self.pos_act(pos)

        opacity = self.opacity_act(opacity)
        scale = self.scale_act(scaling)
        rotation = self.rot_act(rotation)
        rgbs = self.rgb_act(rgb)

        gaussians = torch.cat(
            [pos, opacity, scale, rotation, rgbs], dim=-1
        )  # [B, N, 14]

        return gaussians
