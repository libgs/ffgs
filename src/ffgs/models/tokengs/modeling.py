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
# Modified by the libgs contributors, 2026, from tokengs/models/tokengs.py of
# https://github.com/nv-tlabs/TokenGS (b16269c): the feed-forward inference path
# (`forward_reconstruction`) as a `PyTorchModelHubMixin` module built from keyword
# arguments. Removed the renderer, losses, LPIPS, test-time tuning, and time /
# dynamic-token conditioning; einops replaced by the equivalent reshape / permute.
# Parameter names are unchanged, so upstream checkpoints load as they are.

"""The TokenGS network: input views -> Gaussians in the model frame."""

from __future__ import annotations

from dataclasses import fields
from functools import partial

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .activations import ClipActivationHead
from .attention import PatchEmbed
from .enc_dec import EncDecBackbone, TokenGSConfig

_CONFIG_FIELDS = frozenset(f.name for f in fields(TokenGSConfig))


class TokenGS(nn.Module, PyTorchModelHubMixin):
    """
    TokenGS model with encoder-decoder architecture: learnable Gaussian tokens
    cross-attend to the encoded input views and decode to 3D Gaussians.

    Input (from `TokenGSProcessor`): {"images": [B, V, 3, H, W] ImageNet-normalised
    RGB, "plucker": [B, V, 6, H, W]}. Output: Gaussians [B, N, 14] in the model
    frame -- xyz, opacity, scale, rotation (wxyz), rgb -- with
    N = num_gs_tokens * dec_patch_size ** 2. The arguments are those of
    `TokenGSConfig`.
    """

    def __init__(
        self,
        patch_size: int = 8,
        dec_patch_size: int | None = None,
        enc_depth: int = 3,
        dec_depth: int = 12,
        enc_embed_dim: int = 1024,
        enc_num_heads: int = 16,
        mlp_ratio: float = 4,
        clip_head_readout_std: float = 0.002,
        clip_head_z_init: float | None = None,
        dec_init_values: float | None = None,
        gaussian_scale_cap: float = 0.075,
        opacity_bias: float = 2.0,
        gaussian_z_offset: float = 1.0,
        num_gs_tokens: int = 1024,
        token_dim: int = 1024,
        gs_token_std: float = 1e-2,
        use_multiscale_encoder: bool = False,
        multiscale_encoder_layers: tuple[int, ...] | list[int] = (5, 7, 9, 11),
        use_latent_bottleneck: bool = False,
        num_latents: int = 4096,
        latent_cross_attn_depth: int = 12,
    ):
        arguments = {k: v for k, v in locals().items() if k in _CONFIG_FIELDS}
        super().__init__()
        self.opt = TokenGSConfig(**arguments)

        norm_layer_factory = partial(nn.LayerNorm, bias=True)

        self.patch_embed = PatchEmbed(
            patch_size=self.opt.patch_size,
            in_chans=3,
            embed_dim=self.opt.enc_embed_dim,
            norm_layer=norm_layer_factory,
        )

        self.patch_plucker_embed = PatchEmbed(
            patch_size=self.opt.patch_size,
            in_chans=6,
            embed_dim=self.opt.enc_embed_dim,
            norm_layer=norm_layer_factory,
        )

        # Encoder-decoder architecture
        self.enc_dec_backbone = EncDecBackbone(self.opt)

        # Activation head (always clip)
        self.activation_head = ClipActivationHead(self.opt)

        # Learnable GS tokens
        self.gs_tokens = nn.Parameter(
            self.opt.gs_token_std
            * torch.randn(self.opt.num_gs_tokens, self.opt.token_dim)
        )

    def _embed_encoder_input(
        self, images_rgb: torch.Tensor, plucker: torch.Tensor
    ) -> torch.Tensor:
        B, V, _, H, W = images_rgb.shape
        height = int(H // self.opt.patch_size)
        width = int(W // self.opt.patch_size)

        assert (
            height * self.opt.patch_size == H
        ), f"H={H} must be divisible by patch_size={self.opt.patch_size}"
        assert (
            width * self.opt.patch_size == W
        ), f"W={W} must be divisible by patch_size={self.opt.patch_size}"

        # Reshape for tokenization
        images_rgb_reshaped = images_rgb.reshape(B * V, 3, H, W)
        plucker_reshaped = plucker.reshape(B * V, 6, H, W)

        # Embed RGB patches
        x = self.patch_embed(images_rgb_reshaped)
        # Add Plucker embeddings
        x_plucker_emb = self.patch_plucker_embed(plucker_reshaped)
        x = x + x_plucker_emb  # B*V, N, C

        # "(b v) n c -> b (v n) c": one sequence per batch
        return x.reshape(B, V * x.shape[1], x.shape[2])

    def forward_encoder(
        self, images_rgb: torch.Tensor, plucker: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input views into keys and values for the decoder's
        cross-attention, each [B, heads, tokens, head_dim]."""
        x = self._embed_encoder_input(images_rgb, plucker)
        return self.enc_dec_backbone._encode_to_kv(x)

    def get_gs_tokens(self, batch_size: int) -> torch.Tensor:
        """Initial GS tokens [B, num_gs_tokens, C]."""
        return self.gs_tokens.unsqueeze(0).expand(batch_size, -1, -1).clone()

    def forward_decoder(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        gs_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Keys / values -> Gaussians [B, N, 14]."""
        if gs_tokens is None:
            gs_tokens = self.get_gs_tokens(batch_size=keys.shape[0])

        for layer in self.enc_dec_backbone.decoder_blocks:
            gs_tokens = layer(gs_tokens=gs_tokens, keys=keys, values=values)

        # Convert to Gaussians
        gaussians = self.activation_head(gs_tokens)

        # give gaussians an offset to the z axis so it is visible when initialized
        gaussians[..., 2] = gaussians[..., 2] + self.opt.gaussian_z_offset

        return gaussians

    def forward(self, model_input: dict[str, torch.Tensor]) -> torch.Tensor:
        keys, values = self.forward_encoder(
            model_input["images"], model_input["plucker"]
        )
        return self.forward_decoder(keys, values)
