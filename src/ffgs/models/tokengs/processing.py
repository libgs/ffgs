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
# Modified by the libgs contributors, 2026: the evaluation-time data path of
# https://github.com/nv-tlabs/TokenGS (b16269c) as an ffgs `Processor` --
# `ImageTransform` and `ray_condition` from tokengs/utils/data.py, camera
# normalisation and input normalisation from tokengs/data/provider.py (`Provider.
# _preprocess`), render settings from tokengs/rendering/gs.py and tokengs/options.py.
# Training-only branches (flips, margins, timestamps, depth-based scaling) removed.

"""TokenGS pre / post-processing: the author's evaluation data path.

Per sample, as `tokengs/data/provider.py` does at evaluation:

1. crop the largest centred region with the aspect ratio of `image_shape`, then
   resize it to `image_shape` (torchvision bilinear, antialiased). The intrinsics
   follow the upstream arithmetic exactly, including its half-pixel offset on odd
   crop margins (the pixels are cropped at the rounded offset, K is shifted by the
   exact one);
2. cameras relative to the first input view (`"first_cam"`) or to the mean input
   camera (`"mean_cam"`), translations times `scene_scale`;
3. model input: ImageNet-normalised images and per-pixel Plücker rays.

The Gaussians come out in that frame (`GSOutput.to_world()` maps them to the
caller's world); colours are RGB (no SH). Rendering uses the upstream background and near / far
planes. `crop_mode="pad"` / `"none"` replace step 1 with the ffgs rules.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from ...geometry import ModelFrame, anchored_frame
from ...image import fit_images, fit_intrinsics, rescale
from ...processor import ImageFitConfig, Prepared, Processor
from ...types import Cameras, Gaussians, Views

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CAMERA_NORMALIZATIONS = ("first_cam", "mean_cam")
_AUTOCAST = {"bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class TokenGSProcessorConfig(ImageFitConfig):
    image_shape: list[int] = field(default_factory=lambda: [256, 448])
    resize_mode: str = "bilinear"
    # "first_cam": poses relative to the first input view; "mean_cam": to the mean
    # input camera (the 2026.6 latent-bottleneck models).
    camera_normalization: str = "first_cam"
    scene_scale: float = 0.15
    # The input view count the checkpoint was trained for. Other counts run (the
    # number of Gaussians does not depend on it) but warn. None: any.
    num_input_views: int | None = None
    image_mean: list[float] = field(default_factory=lambda: list(IMAGENET_MEAN))
    image_std: list[float] = field(default_factory=lambda: list(IMAGENET_STD))
    # Render settings of the upstream renderer, in model-frame units.
    znear: float = 0.025
    zfar: float = 125.0
    background: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    # Autocast dtype of the model on CUDA ("bfloat16", as upstream evaluates, or
    # "float16"); None runs in fp32.
    autocast: str | None = "bfloat16"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.camera_normalization not in CAMERA_NORMALIZATIONS:
            raise ValueError(
                f"camera_normalization must be one of {CAMERA_NORMALIZATIONS}"
            )
        if self.autocast is not None and self.autocast not in _AUTOCAST:
            raise ValueError(f"autocast must be None or one of {sorted(_AUTOCAST)}")


class TokenGSProcessor(Processor):
    config_cls = TokenGSProcessorConfig

    def preprocess(
        self, views: Views, device: torch.device, scene_scale: float | None = None
    ) -> Prepared:
        # TokenGS is conditioned on the input cameras (Plücker rays).
        views.require("c2w", "intrinsics", by="TokenGS")
        cfg = self.config
        num_views = views.images.shape[1]
        if cfg.num_input_views is not None and num_views != cfg.num_input_views:
            warnings.warn(
                f"this TokenGS checkpoint was trained for {cfg.num_input_views} "
                f"input views, got {num_views}",
                stacklevel=3,
            )
        scale = cfg.scene_scale if scene_scale is None else scene_scale
        frame = self.frame(views.c2w, scale)
        images, k = self.fit_pixel_views(views)
        c2w = frame.c2w_to_model(views.c2w.float())
        h, w = images.shape[-2:]
        plucker = torch.stack(
            [
                ray_condition(k[i : i + 1], c2w[i : i + 1], h, w)[0]
                for i in range(len(k))
            ]
        )
        normalized = torch.stack(
            [TF.normalize(x, cfg.image_mean, cfg.image_std) for x in images]
        )
        model_input = {"images": normalized.to(device), "plucker": plucker.to(device)}
        return Prepared(model_input, frame)

    def frame(self, c2w: torch.Tensor, scale: float) -> ModelFrame:
        """The model frame of input views c2w [B, V, 4, 4]."""
        if self.config.camera_normalization == "first_cam":
            return anchored_frame(c2w, scale)
        anchor = torch.stack([mean_camera(x.float()) for x in c2w])
        return ModelFrame(anchor_c2w=anchor, scale=float(scale))

    def postprocess(self, model_output: torch.Tensor) -> Gaussians:
        g = model_output.float()
        return Gaussians(
            means=g[..., 0:3],
            opacities=g[..., 3],
            scales=g[..., 4:7],
            quats=g[..., 7:11],
            colors=g[..., 11:14],
            sh_degree=None,
        )

    def render_planes(self, frame: ModelFrame) -> tuple[float, float]:
        return self.config.znear / frame.scale, self.config.zfar / frame.scale

    @property
    def background(self) -> tuple[float, float, float]:
        return tuple(self.config.background)

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        return None if self.config.autocast is None else _AUTOCAST[self.config.autocast]

    # --- fitting inputs ---------------------------------------------------------

    def fit_pixel_views(
        self, views: Views, image_shape: list[int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Images [B, V, 3, h, w] and pixel intrinsics [B, V, 4] (fx, fy, cx, cy)
        at `image_shape` (default: the model's), by the config's `crop_mode`."""
        views.require("intrinsics", by="TokenGS")
        cfg = self.config
        shape = tuple(image_shape or cfg.image_shape)
        src_h, src_w = views.image_shape
        if views.normalized_intrinsics:
            k = _pixel_k(views.intrinsics.float(), (src_h, src_w))
        else:
            k = _flat_k(views.intrinsics.float())
        images, ks = [], []
        for img, ki in zip(views.images, k):
            if cfg.crop_mode == "crop":
                img, ki = crop_and_resize(img, ki, shape, self._resize)
            else:
                img, kn = fit_images(
                    img,
                    _normalized_k(ki, (src_h, src_w)),
                    shape,
                    resize_mode=cfg.resize_mode,
                    crop_mode=cfg.crop_mode,
                    resize=self.resize,
                    pad_value=cfg.pad_value,
                )
                ki = _pixel_k(kn, shape)
            images.append(img)
            ks.append(ki)
        return torch.stack(images), torch.stack(ks)

    def fit_views(self, views: Views, image_shape: list[int] | None = None) -> Views:
        images, k = self.fit_pixel_views(views, image_shape)
        return Views(
            images,
            _normalized_k(k, tuple(images.shape[-2:])),
            views.c2w,
            normalized_intrinsics=True,
        )

    def fit_cameras(
        self, cameras: Cameras, image_shape: list[int] | None = None
    ) -> Cameras:
        cfg = self.config
        shape = tuple(image_shape or cfg.image_shape)
        if cfg.crop_mode != "crop":
            k = fit_intrinsics(
                cameras.normalized_k, cameras.image_shape, shape, cfg.crop_mode
            )
            return Cameras(cameras.c2w, k, shape, normalized_intrinsics=True)
        if cameras.normalized_intrinsics:
            k = _pixel_k(cameras.intrinsics.float(), cameras.image_shape)
        else:
            k = _flat_k(cameras.intrinsics.float())
        k = crop_and_resize_intrinsics(k, cameras.image_shape, shape)
        return Cameras(
            cameras.c2w, _normalized_k(k, shape), shape, normalized_intrinsics=True
        )

    def _resize(self, images: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        if self.resize is not None:
            return self.resize(images, shape)
        return rescale(images, shape, self.config.resize_mode)


def crop_region(
    image_shape: tuple[int, int], shape: tuple[int, int]
) -> tuple[int, int]:
    """(h, w) of the largest centred crop of `image_shape` with the aspect ratio
    of `shape` (upstream `ImageTransform` with `max_crop=True`)."""
    ori_h, ori_w = image_shape
    crop_ratio = min(ori_h / shape[0], ori_w / shape[1])
    return int(shape[0] * crop_ratio), int(shape[1] * crop_ratio)


def crop_and_resize_intrinsics(
    intrinsics: torch.Tensor, image_shape: tuple[int, int], shape: tuple[int, int]
) -> torch.Tensor:
    """Pixel intrinsics [..., 4] (fx, fy, cx, cy) through `crop_and_resize`."""
    ori_h, ori_w = image_shape
    new_h, new_w = crop_region(image_shape, shape)
    # u,v convention: shift relative to original (un-cropped) raster
    shift = ((new_w - ori_w) / 2, (new_h - ori_h) / 2)
    scale = (shape[1] / new_w, shape[0] / new_h)
    return torch.stack(
        [
            intrinsics[..., 0] * scale[0],
            intrinsics[..., 1] * scale[1],
            (intrinsics[..., 2] + shift[0]) * scale[0],
            (intrinsics[..., 3] + shift[1]) * scale[1],
        ],
        dim=-1,
    )


def crop_and_resize(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    shape: tuple[int, int],
    resize=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Centre-crop [V, 3, H, W] images to the aspect ratio of `shape`, resize to
    `shape` (default: torchvision bilinear, antialiased), and map the pixel
    intrinsics [V, 4] alike."""
    image_shape = tuple(images.shape[-2:])
    images = TF.center_crop(images, list(crop_region(image_shape, shape)))
    if resize is None:
        images = TF.resize(
            images, list(shape), TF.InterpolationMode.BILINEAR, antialias=True
        )
    else:
        images = resize(images, tuple(shape))
    return images, crop_and_resize_intrinsics(intrinsics, image_shape, shape)


def mean_camera(c2w: torch.Tensor) -> torch.Tensor:
    """The mean camera [4, 4] of input views c2w [V, 4, 4]: mean position, mean
    forward / down axes orthonormalised (upstream `_normalize_camera_mean_cam`)."""
    position_avg = c2w[:, :3, 3].mean(0)  # (3,)
    forward_avg = c2w[:, :3, 2].mean(0)  # (3,)
    down_avg = c2w[:, :3, 1].mean(0)  # (3,)
    # gram-schmidt process
    forward_avg = F.normalize(forward_avg, dim=0)
    down_avg = F.normalize(down_avg - down_avg.dot(forward_avg) * forward_avg, dim=0)
    right_avg = torch.linalg.cross(down_avg, forward_avg)
    pos_avg = torch.stack([right_avg, down_avg, forward_avg, position_avg], dim=1)
    bottom = torch.tensor([[0, 0, 0, 1]], device=pos_avg.device).float()
    return torch.cat([pos_avg, bottom], dim=0)  # (4, 4)


def ray_condition(K: torch.Tensor, c2w: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Plücker rays (o x d, d) [V, 6, H, W] through pixel centres, for pixel
    intrinsics K [1, V, 4] and model-frame c2w [1, V, 4, 4]."""
    B, V = K.shape[:2]
    j, i = torch.meshgrid(
        torch.linspace(0, H - 1, H, device=K.device, dtype=K.dtype),
        torch.linspace(0, W - 1, W, device=K.device, dtype=K.dtype),
        indexing="ij",
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5

    fx, fy, cx, cy = K.chunk(4, dim=-1)
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)
    rays_o = c2w[..., :3, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d)

    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    return plucker.reshape(B, V, H, W, 6).permute(0, 1, 4, 2, 3).contiguous()


def _flat_k(k: torch.Tensor) -> torch.Tensor:
    """3x3 K [..., 3, 3] -> [..., 4] (fx, fy, cx, cy)."""
    return torch.stack((k[..., 0, 0], k[..., 1, 1], k[..., 0, 2], k[..., 1, 2]), -1)


def _pixel_k(k: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    """Normalised 3x3 K -> pixel [..., 4] (fx, fy, cx, cy)."""
    h, w = image_shape
    return torch.stack(
        (k[..., 0, 0] * w, k[..., 1, 1] * h, k[..., 0, 2] * w, k[..., 1, 2] * h), -1
    )


def _normalized_k(k: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    """Pixel [..., 4] (fx, fy, cx, cy) -> normalised 3x3 K."""
    h, w = image_shape
    out = torch.zeros(*k.shape[:-1], 3, 3, dtype=k.dtype, device=k.device)
    out[..., 0, 0] = k[..., 0] / w
    out[..., 1, 1] = k[..., 1] / h
    out[..., 0, 2] = k[..., 2] / w
    out[..., 1, 2] = k[..., 3] / h
    out[..., 2, 2] = 1.0
    return out
