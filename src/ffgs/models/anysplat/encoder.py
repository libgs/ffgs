# Copyright (c) 2025 Lihan Jiang and Yucheng Mao
# SPDX-License-Identifier: MIT (see the AnySplat section of LICENSE, this directory)
#
# Modified by the libgs contributors, 2026, from src/model/encoder/anysplat.py and
# src/model/encoder/encoder.py of https://github.com/InternRobotics/AnySplat
# (5f5e208): inference-only subset of `EncoderAnySplat` (pred_head_type="depth",
# pose_free=True, the released configuration).
# - The VGGT modules are built directly from the architecture arguments instead of
#   `VGGT.from_pretrained("facebook/VGGT-1B")` (whose weights the AnySplat checkpoint
#   replaces anyway); the point / track heads it also built are not. The Gaussian
#   head's `dim_in` is `2 * embed_dim` (upstream: the literal 2048, VGGT-1B's).
# - torch_scatter's `scatter_max` / `scatter_add` replaced by the same reductions in
#   torch (`scatter_reduce_("amax")`; `scatter_add_` on a broadcast index, which is
#   what torch_scatter's `scatter_add` runs).
# - The aggregated tokens are made float32 before the float32 heads: a no-op on
#   CUDA, where the bfloat16 autocast already returns them in float32, and what lets
#   the model run on other devices, where that autocast is inactive.
# - `infos["num_gaussians"]` records each sample's Gaussian count: for batches,
#   upstream pads the samples to the largest count (`pad_tensor_list`).
# - Removed: distillation, parameter freezing, `normalize_pts3d` /
#   `align_pts_all_with_pts3d`, the point-head branch, visualization dumps, the data
#   shim, the `print`s and the unconditional `torch.cuda.empty_cache()`; einops /
#   jaxtyping no longer needed. `EncoderOutput` keeps the fields used here.

"""AnySplat's encoder: VGGT aggregator, camera and depth heads, a DPT head for the
Gaussian parameters, and voxel fusion of the per-pixel Gaussians."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from .gaussian_adapter import GaussianAdapterCfg, Gaussians, UnifiedGaussianAdapter
from .gs_head import VGGT_DPT_GS_Head
from .vggt.heads.camera_head import CameraHead
from .vggt.heads.dpt_head import DPTHead
from .vggt.models.aggregator import Aggregator
from .vggt.utils.geometry import batchify_unproject_depth_map_to_point_map
from .vggt.utils.pose_enc import pose_encoding_to_extri_intri

inf = float("inf")


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class GSHeadParams:
    dec_depth: int = 23
    patch_size: tuple[int, int] = (14, 14)
    enc_embed_dim: int = 2048
    dec_embed_dim: int = 2048
    feature_dim: int = 256
    depth_mode = ("exp", -inf, inf)
    conf_mode = True


@dataclass
class EncoderAnySplatCfg:
    """The inference fields of upstream's `EncoderAnySplatCfg` (same names and
    defaults), and the VGGT architecture (ffgs: upstream fixes it to VGGT-1B)."""

    voxel_size: float
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    gs_prune: bool = False
    opacity_threshold: float = 0.001
    gs_keep_ratio: float = 1.0
    render_conf: bool = False
    opacity_conf: bool = False
    conf_threshold: float = 0.1
    intermediate_layer_idx: list[int] | None = None
    voxelize: bool = False
    # VGGT-1B: `VGGT(img_size=518, patch_size=14, embed_dim=1024)`; extra keyword
    # arguments of each module (all empty for the released model).
    img_size: int = 518
    patch_size: int = 14
    embed_dim: int = 1024
    aggregator: dict[str, Any] = field(default_factory=dict)
    camera_head: dict[str, Any] = field(default_factory=dict)
    depth_head: dict[str, Any] = field(default_factory=dict)
    gs_head: dict[str, Any] = field(default_factory=dict)


@dataclass
class EncoderOutput:
    gaussians: Gaussians
    pred_pose_enc_list: list[Tensor] | None
    pred_context_pose: dict | None
    depth_dict: dict | None
    infos: dict | None


class EncoderAnySplat(nn.Module):
    def __init__(self, cfg: EncoderAnySplatCfg) -> None:
        super().__init__()
        self.cfg = cfg
        dim_in = 2 * cfg.embed_dim
        self.aggregator = Aggregator(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            **cfg.aggregator,
        ).to(torch.bfloat16)
        self.camera_head = CameraHead(dim_in=dim_in, **cfg.camera_head)
        self.depth_head = DPTHead(
            dim_in=dim_in,
            output_dim=2,
            activation="exp",
            conf_activation="expp1",
            **cfg.depth_head,
        )

        self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)

        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity
        self.voxel_size = cfg.voxel_size
        # fake backbone for head parameters
        head_params = GSHeadParams()
        self.gaussian_param_head = VGGT_DPT_GS_Head(
            dim_in=dim_in,
            patch_size=head_params.patch_size,
            output_dim=self.raw_gs_dim + 1,
            activation="norm_exp",
            conf_activation="expp1",
            features=head_params.feature_dim,
            **cfg.gs_head,
        )

    def map_pdf_to_opacity(
        self,
        pdf: Tensor,
        global_step: int,
    ) -> Tensor:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def pad_tensor_list(self, tensor_list, pad_shape, value=0.0):
        padded = []
        for t in tensor_list:
            pad_len = pad_shape[0] - t.shape[0]
            if pad_len > 0:
                padding = torch.full(
                    (pad_len, *t.shape[1:]), value, device=t.device, dtype=t.dtype
                )
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        return torch.stack(padded)

    def voxelizaton_with_fusion(self, img_feat, pts3d, voxel_size, conf=None):
        # img_feat: B*V, C, H, W
        # pts3d: B*V, 3, H, W
        V, C, H, W = img_feat.shape
        pts3d_flatten = pts3d.permute(0, 2, 3, 1).flatten(0, 2)

        voxel_indices = (pts3d_flatten / voxel_size).round().int()  # [B*V*N, 3]
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_indices, dim=0, return_inverse=True, return_counts=True
        )

        # Flatten confidence scores and features
        conf_flat = conf.flatten()  # [B*V*N]
        anchor_feats_flat = img_feat.permute(0, 2, 3, 1).flatten(0, 2)  # [B*V*N, ...]

        # Compute softmax weights per voxel
        conf_voxel_max = _scatter_max(conf_flat, inverse_indices, len(unique_voxels))
        conf_exp = torch.exp(conf_flat - conf_voxel_max[inverse_indices])
        voxel_weights = _scatter_add(
            conf_exp, inverse_indices, len(unique_voxels)
        )  # [num_unique_voxels]
        weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(
            -1
        )  # [B*V*N, 1]

        # Compute weighted average of positions and features
        weighted_pts = pts3d_flatten * weights
        weighted_feats = anchor_feats_flat.squeeze(1) * weights

        # Aggregate per voxel
        voxel_pts = _scatter_add(
            weighted_pts, inverse_indices, len(unique_voxels)
        )  # [num_unique_voxels, 3]
        voxel_feats = _scatter_add(
            weighted_feats, inverse_indices, len(unique_voxels)
        )  # [num_unique_voxels, feat_dim]

        return voxel_pts, voxel_feats

    def forward(
        self,
        image: torch.Tensor,
        global_step: int = 0,
    ) -> EncoderOutput:
        device = image.device
        b, v, _, h, w = image.shape

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            aggregated_tokens_list, patch_start_idx = self.aggregator(
                image.to(torch.bfloat16),
                intermediate_layer_idx=self.cfg.intermediate_layer_idx,
            )
        aggregated_tokens_list = [t.float() for t in aggregated_tokens_list]

        with torch.amp.autocast("cuda", enabled=False):
            pred_pose_enc_list = self.camera_head(aggregated_tokens_list)
            last_pred_pose_enc = pred_pose_enc_list[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                last_pred_pose_enc, image.shape[-2:]
            )  # only for debug

            depth_map, depth_conf = self.depth_head(
                aggregated_tokens_list,
                images=image,
                patch_start_idx=patch_start_idx,
            )
            pts_all = batchify_unproject_depth_map_to_point_map(
                depth_map, extrinsic, intrinsic
            )

            if self.cfg.render_conf:
                conf_valid = torch.quantile(
                    depth_conf.flatten(0, 1), self.cfg.conf_threshold
                )
                conf_valid_mask = depth_conf > conf_valid
            else:
                conf_valid_mask = torch.ones_like(depth_conf, dtype=torch.bool)

        # dpt style gs_head input format
        out = self.gaussian_param_head(
            aggregated_tokens_list,
            pts_all.flatten(0, 1).permute(0, 3, 1, 2),
            image,
            patch_start_idx=patch_start_idx,
            image_size=(h, w),
        )

        del aggregated_tokens_list, patch_start_idx

        pts_flat = pts_all.flatten(2, 3)
        scene_scale = pts_flat.norm(dim=-1).mean().clip(min=1e-8)

        anchor_feats, conf = out[:, :, : self.raw_gs_dim], out[:, :, self.raw_gs_dim]

        neural_feats_list, neural_pts_list = [], []
        if self.cfg.voxelize:
            for b_i in range(b):
                neural_pts, neural_feats = self.voxelizaton_with_fusion(
                    anchor_feats[b_i],
                    pts_all[b_i].permute(0, 3, 1, 2).contiguous(),
                    self.voxel_size,
                    conf=conf[b_i],
                )
                neural_feats_list.append(neural_feats)
                neural_pts_list.append(neural_pts)
        else:
            for b_i in range(b):
                neural_feats_list.append(
                    anchor_feats[b_i].permute(0, 2, 3, 1)[conf_valid_mask[b_i]]
                )
                neural_pts_list.append(pts_all[b_i][conf_valid_mask[b_i]])

        # ffgs: the per-sample Gaussian counts, before padding to the batch maximum.
        num_gaussians = [f.shape[0] for f in neural_feats_list]
        max_voxels = max(f.shape[0] for f in neural_feats_list)
        neural_feats = self.pad_tensor_list(
            neural_feats_list, (max_voxels,), value=-1e10
        )

        neural_pts = self.pad_tensor_list(
            neural_pts_list, (max_voxels,), -1e4
        )  # -1 == invalid voxel

        depths = neural_pts[..., -1].unsqueeze(-1)
        densities = neural_feats[..., 0].sigmoid()

        assert len(densities.shape) == 2, "the shape of densities should be (B, N)"
        assert neural_pts.shape[1] > 1, "the number of voxels should be greater than 1"

        opacity = self.map_pdf_to_opacity(densities, global_step).squeeze(-1)
        if self.cfg.opacity_conf:
            shift = torch.quantile(depth_conf, self.cfg.conf_threshold)
            opacity = opacity * torch.sigmoid(depth_conf - shift)[
                conf_valid_mask
            ].unsqueeze(
                0
            )  # little bit hacky

        # GS Prune, but only works when bs = 1
        # if want to support bs > 1, need to random prune gaussians based on the rank of opacity like LongLRM
        # Note: we not prune gaussians here, but we will try it in the future
        if self.cfg.gs_prune and b == 1:
            opacity_threshold = self.cfg.opacity_threshold
            gaussian_usage = opacity > opacity_threshold  # (B, N)

            if (gaussian_usage.sum() / gaussian_usage.numel()) > self.cfg.gs_keep_ratio:
                # rank by opacity
                num_keep = int(gaussian_usage.shape[1] * self.cfg.gs_keep_ratio)
                idx_sort = opacity.argsort(dim=1, descending=True)
                keep_idx = idx_sort[:, :num_keep]
                gaussian_usage = torch.zeros_like(gaussian_usage, dtype=torch.bool)
                gaussian_usage.scatter_(1, keep_idx, True)

            neural_pts = neural_pts[gaussian_usage].view(b, -1, 3).contiguous()
            depths = depths[gaussian_usage].view(b, -1, 1).contiguous()
            neural_feats = (
                neural_feats[gaussian_usage].view(b, -1, self.raw_gs_dim).contiguous()
            )
            opacity = opacity[gaussian_usage].view(b, -1).contiguous()

        gaussians = self.gaussian_adapter.forward(
            neural_pts,
            depths,
            opacity,
            neural_feats[..., 1:].squeeze(2),
        )

        infos = {}
        infos["scene_scale"] = scene_scale
        infos["voxelize_ratio"] = densities.shape[1] / (h * w * v)
        if self.cfg.gs_prune and b == 1:
            num_gaussians = [neural_pts.shape[1]]
        infos["num_gaussians"] = torch.tensor(num_gaussians, device=device)

        extrinsic_padding = (
            torch.tensor([0, 0, 0, 1], device=device, dtype=extrinsic.dtype)
            .view(1, 1, 1, 4)
            .repeat(b, v, 1, 1)
        )
        intrinsic = intrinsic.clone()  # Create a new tensor
        intrinsic = torch.stack(
            [intrinsic[:, :, 0] / w, intrinsic[:, :, 1] / h, intrinsic[:, :, 2]], dim=2
        )

        return EncoderOutput(
            gaussians=gaussians,
            pred_pose_enc_list=pred_pose_enc_list,
            pred_context_pose=dict(
                extrinsic=torch.cat([extrinsic, extrinsic_padding], dim=2).inverse(),
                intrinsic=intrinsic,
            ),
            depth_dict=dict(depth=depth_map, conf_valid_mask=conf_valid_mask),
            infos=infos,
        )


def _scatter_max(src: Tensor, index: Tensor, size: int) -> Tensor:
    """torch_scatter's `scatter_max(src, index, dim=0)[0]` for 1-D `src` whose
    `index` hits every one of `size` slots (the maximum is exact in any order)."""
    out = src.new_empty(size)
    return out.scatter_reduce_(0, index, src, "amax", include_self=False)


def _scatter_add(src: Tensor, index: Tensor, size: int) -> Tensor:
    """torch_scatter's `scatter_add(src, index, dim=0)`: its implementation, a
    `scatter_add_` into zeros along an index broadcast to `src`."""
    index = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out = src.new_zeros(size, *src.shape[1:])
    return out.scatter_add_(0, index, src)
