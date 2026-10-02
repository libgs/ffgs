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
# Modified by the libgs contributors, 2026, from tokengs/models/enc_dec.py of
# https://github.com/nv-tlabs/TokenGS (b16269c): inference-only subset. Takes a
# `TokenGSConfig` instead of the training `Options`; einops replaced by the
# equivalent reshape / permute; dynamic Gaussian tokens (block-causal
# FlexAttention) removed. Parameter names are unchanged.

"""TokenGS encoder-decoder backbone: ViT encoder, optional latent bottleneck, and a
decoder whose Gaussian tokens cross-attend to the encoder output."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention, Block, LayerScale, Mlp


@dataclass
class TokenGSConfig:
    """Architecture of a TokenGS checkpoint (the model part of the upstream
    `Options`, same names and defaults)."""

    patch_size: int = 8
    dec_patch_size: int | None = None
    enc_depth: int = 3
    dec_depth: int = 12
    enc_embed_dim: int = 1024
    enc_num_heads: int = 16
    mlp_ratio: float = 4
    clip_head_readout_std: float = 0.002
    clip_head_z_init: float | None = None
    dec_init_values: float | None = None
    gaussian_scale_cap: float = 0.075
    opacity_bias: float = 2.0
    gaussian_z_offset: float = 1.0
    num_gs_tokens: int = 1024
    token_dim: int = 1024
    gs_token_std: float = 1e-2
    use_multiscale_encoder: bool = False
    multiscale_encoder_layers: tuple[int, ...] = (5, 7, 9, 11)
    use_latent_bottleneck: bool = False
    num_latents: int = 4096
    latent_cross_attn_depth: int = 12

    def __post_init__(self) -> None:
        if self.dec_patch_size is None:
            self.dec_patch_size = self.patch_size
        self.multiscale_encoder_layers = tuple(self.multiscale_encoder_layers)


def _split_kv(x: torch.Tensor, num_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
    """ "b n (kv h c) -> kv b h n c", unbound into (k, v)."""
    b, n, _ = x.shape
    x = x.reshape(b, n, 2, num_heads, -1).permute(2, 0, 3, 1, 4)
    return x[0], x[1]


class DecoderBlock(nn.Module):
    """
    Decoder block for encoder-decoder architecture.
    Works like a transformer decoder layer except the keys/values are provided by
    the encoder (already normalized).
    """

    class SelfAttnBlock(nn.Module):
        def __init__(self, dim: int, num_heads: int, qkv_bias: bool, qk_norm: bool):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.gs_self_attn = Attention(
                dim, num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm
            )

        def forward(self, gs_tokens: torch.Tensor) -> torch.Tensor:
            queries_normed = self.norm(gs_tokens)
            return self.gs_self_attn(queries_normed)

    class CrossAttnBlock(nn.Module):
        def __init__(self, dim: int, num_heads: int, qkv_bias: bool, q_norm: bool):
            super().__init__()
            self.num_heads = num_heads
            self.gs_token_norm = nn.LayerNorm(dim)
            self.q_norm = nn.LayerNorm(dim // num_heads) if q_norm else nn.Identity()
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.out_proj = nn.Linear(dim, dim)

        def forward(
            self, gs_tokens: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
        ) -> torch.Tensor:
            gs_tokens_normed = self.gs_token_norm(gs_tokens)
            q = self.q_proj(gs_tokens_normed)
            b, n, _ = q.shape
            # "b n (h d) -> b h n d"
            q = q.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            q = self.q_norm(q)

            cross_attn_output = F.scaled_dot_product_attention(q, keys, values)
            # "b h n d -> b n (h d)"
            cross_attn_output = cross_attn_output.permute(0, 2, 1, 3).reshape(b, n, -1)
            return self.out_proj(cross_attn_output)

    class MlpBlock(nn.Module):
        def __init__(self, dim: int, mlp_ratio: float, ffn_bias: bool):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, bias=ffn_bias)

        def forward(self, gs_tokens: torch.Tensor) -> torch.Tensor:
            return self.mlp(self.norm(gs_tokens))

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        ffn_bias: bool,
        qk_norm: bool,
        init_values: float | None = None,
    ):
        super().__init__()

        def make_scale() -> nn.Module:
            return (
                LayerScale(dim, init_values=init_values)
                if init_values
                else nn.Identity()
            )

        self.gs_self_attn = DecoderBlock.SelfAttnBlock(
            dim, num_heads, qkv_bias, qk_norm=qk_norm
        )
        self.gs_self_attn_scale = make_scale()

        self.gs_cross_attn = DecoderBlock.CrossAttnBlock(
            dim, num_heads, qkv_bias, q_norm=qk_norm
        )
        self.gs_cross_attn_scale = make_scale()

        self.mlp = DecoderBlock.MlpBlock(dim, mlp_ratio, ffn_bias)
        self.mlp_scale = make_scale()

    def forward(
        self, gs_tokens: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        gs_tokens = gs_tokens + self.gs_cross_attn_scale(
            self.gs_cross_attn(gs_tokens, keys, values)
        )
        gs_tokens = gs_tokens + self.gs_self_attn_scale(self.gs_self_attn(gs_tokens))
        gs_tokens = gs_tokens + self.mlp_scale(self.mlp(gs_tokens))

        return gs_tokens


class EncDecBackbone(nn.Module):
    """
    An encoder-decoder backbone for the EncDec architecture.

    Encoder: a stack of ViT blocks which produce a latent representation. This is
    followed by a key-value projection to produce a key and value for the decoder.

    Decoder: a stack of transformer decoder layers which attend from GS tokens to
    the encoder output and among themselves.
    """

    def __init__(self, opt: TokenGSConfig):
        super().__init__()

        self.opt = opt

        self.encoder = nn.Sequential(
            *[
                Block(
                    self.opt.enc_embed_dim,
                    self.opt.enc_num_heads,
                    self.opt.mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=0.01,
                    qk_norm=True,
                )
                for _ in range(self.opt.enc_depth)
            ]
        )

        self.use_multiscale = self.opt.use_multiscale_encoder
        self.use_latent_bottleneck = self.opt.use_latent_bottleneck

        if self.use_multiscale:
            self.multiscale_layers = self.opt.multiscale_encoder_layers
            self.multiscale_norms = nn.ModuleList(
                [nn.LayerNorm(self.opt.enc_embed_dim) for _ in self.multiscale_layers]
            )
            encoder_feature_dim = self.opt.enc_embed_dim * len(self.multiscale_layers)
        else:
            self.encoder_norm = nn.LayerNorm(self.opt.enc_embed_dim)
            encoder_feature_dim = self.opt.enc_embed_dim

        if not self.use_latent_bottleneck:
            self.kv_proj = nn.Linear(
                encoder_feature_dim, self.opt.enc_embed_dim * 2, bias=True
            )

        self.k_proj_norm = nn.LayerNorm(
            self.opt.enc_embed_dim // self.opt.enc_num_heads
        )

        if self.use_latent_bottleneck:
            self.latents = nn.Parameter(
                torch.randn(self.opt.num_latents, self.opt.enc_embed_dim) * 0.02
            )

            self.latent_feature_kv_proj = nn.Linear(
                encoder_feature_dim, self.opt.enc_embed_dim * 2, bias=True
            )
            self.latent_feature_k_norm = nn.LayerNorm(
                self.opt.enc_embed_dim // self.opt.enc_num_heads
            )
            self.latent_blocks = nn.ModuleList(
                [
                    DecoderBlock(
                        self.opt.enc_embed_dim,
                        self.opt.enc_num_heads,
                        self.opt.mlp_ratio,
                        qkv_bias=True,
                        ffn_bias=True,
                        init_values=0.01,
                        qk_norm=True,
                    )
                    for _ in range(self.opt.latent_cross_attn_depth)
                ]
            )
            self.latent_kv_proj = nn.Linear(
                self.opt.enc_embed_dim, self.opt.enc_embed_dim * 2, bias=True
            )

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    self.opt.enc_embed_dim,
                    self.opt.enc_num_heads,
                    self.opt.mlp_ratio,
                    qkv_bias=True,
                    ffn_bias=True,
                    init_values=(
                        self.opt.dec_init_values
                        if self.opt.dec_init_values is not None
                        else 5e-3 * self.opt.gs_token_std
                    ),
                    qk_norm=True,
                )
                for _ in range(self.opt.dec_depth)
            ]
        )

    def _encode_features(self, image_features: torch.Tensor) -> torch.Tensor:
        if self.use_multiscale:
            x = image_features
            features = []
            for i, block in enumerate(self.encoder):
                x = block(x)
                if i in self.multiscale_layers:
                    features.append(self.multiscale_norms[len(features)](x))
            return torch.cat(features, dim=-1)
        image_features = self.encoder(image_features)
        return self.encoder_norm(image_features)

    def _features_to_kv(
        self, image_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_feature_keys, image_feature_values = _split_kv(
            self.kv_proj(image_features), self.opt.enc_num_heads
        )
        image_feature_keys = self.k_proj_norm(image_feature_keys)
        return image_feature_keys, image_feature_values

    def _features_to_scene_latents(self, image_features: torch.Tensor) -> torch.Tensor:
        feat_keys, feat_values = _split_kv(
            self.latent_feature_kv_proj(image_features), self.opt.enc_num_heads
        )
        feat_keys = self.latent_feature_k_norm(feat_keys)

        latents = self.latents.unsqueeze(0).expand(image_features.shape[0], -1, -1)
        for block in self.latent_blocks:
            latents = block(gs_tokens=latents, keys=feat_keys, values=feat_values)

        return latents

    def _latents_to_kv(
        self, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_keys, latent_values = _split_kv(
            self.latent_kv_proj(latents), self.opt.enc_num_heads
        )
        latent_keys = self.k_proj_norm(latent_keys)
        return latent_keys, latent_values

    def _encode_to_kv(
        self, image_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self._encode_features(image_features)
        if self.use_latent_bottleneck:
            latents = self._features_to_scene_latents(image_features)
            return self._latents_to_kv(latents)
        return self._features_to_kv(image_features)
