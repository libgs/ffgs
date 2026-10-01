"""AnySplat pre / post-processing: the author's inference data path.

Per sample, as `src/utils/image.py:process_image` and the callers in
`src/eval_nvs.py` / `inference.py` of https://github.com/InternRobotics/AnySplat
do:

1. the image is read as 8 bit (inputs are quantised to 8 bit here), resized with
   PIL's bicubic filter so that it covers `image_shape`, the covering side by the
   upstream arithmetic (`int`, not `round`), then centre-cropped to `image_shape`
   ([448, 448] for the released model);
2. `ToTensor() * 2 - 1`, then `(x + 1) * 0.5`: the [0, 1] images the model takes,
   with the float32 rounding of that round trip.

The model needs no cameras: it predicts them (`predicted_cameras`), in the frame of
its first input camera at its own scale. With poses, the pipeline fits that frame
to them (`ModelFrame.fit`). Gaussians come out with SH of degree 4.

Rendering uses upstream's settings (its `DecoderSplattingCUDA`): near plane 1e-10
and gsplat's default far plane, in model units, `radius_clip=0.1`, classic
rasterisation, white background, colours clamped to [0, 1].

`crop_mode="pad"` / `"none"` replace step 1's crop with the ffgs rules; the
`resize_mode` `"bicubic"` (the default) is step 1's PIL kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from ...geometry import ModelFrame
from ...image import CROP_MODES, RESIZE_MODES, center_crop, fit_images, rescale
from ...processor import ImageFitConfig, Prepared, Processor
from ...types import Cameras, Gaussians, Views
from .encoder import EncoderOutput

# The ffgs kernels, and upstream's (PIL bicubic on 8-bit images).
ANYSPLAT_RESIZE_MODES = ("bicubic", *RESIZE_MODES)
RASTERIZE_MODES = ("classic", "antialiased")
PATCH_SIZE = 14


@dataclass
class AnySplatProcessorConfig(ImageFitConfig):
    # [h, w], multiples of the 14-pixel patches; upstream: 448 x 448.
    image_shape: list[int] = field(default_factory=lambda: [448, 448])
    resize_mode: str = "bicubic"
    # Render settings of the upstream renderer; planes in model units.
    znear: float = 1e-10
    zfar: float = 1e10
    background: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    radius_clip: float = 0.1
    rasterize_mode: str = "classic"
    clamp: bool = True

    def __post_init__(self) -> None:
        # ImageFitConfig's checks, with upstream's kernel among the resize modes.
        self.image_shape = [int(x) for x in self.image_shape]
        if len(self.image_shape) != 2:
            raise ValueError(f"image_shape must be [h, w], got {self.image_shape}")
        if any(x <= 0 or x % PATCH_SIZE for x in self.image_shape):
            raise ValueError(
                f"image_shape must be positive multiples of {PATCH_SIZE}, got "
                f"{self.image_shape}"
            )
        if self.resize_mode not in ANYSPLAT_RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {ANYSPLAT_RESIZE_MODES}")
        if self.crop_mode not in CROP_MODES:
            raise ValueError(f"crop_mode must be one of {CROP_MODES}")
        if self.rasterize_mode not in RASTERIZE_MODES:
            raise ValueError(f"rasterize_mode must be one of {RASTERIZE_MODES}")


class AnySplatProcessor(Processor):
    config_cls = AnySplatProcessorConfig

    def preprocess(self, views: Views, device: torch.device) -> Prepared:
        images = self.fit_views(views).images
        # process_image's `ToTensor() * 2 - 1`, undone by its callers.
        images = (images * 2.0 - 1.0 + 1) * 0.5
        # No frame yet: it is fitted to the predicted cameras (`output_frame`).
        return Prepared(images.to(device), None)

    def postprocess(self, model_output: EncoderOutput) -> Gaussians:
        g = model_output.gaussians
        d_sh = g.harmonics.shape[-1]
        gaussians = Gaussians(
            means=g.means.float(),
            scales=g.scales.float(),
            quats=g.rotations[..., [3, 0, 1, 2]].float(),  # xyzw -> wxyz
            opacities=g.opacities.float(),
            colors=g.harmonics.transpose(-1, -2).float(),  # [B, N, d_sh, 3]
            sh_degree=round(d_sh**0.5) - 1,
        )
        counts = (model_output.infos or {}).get("num_gaussians")
        return gaussians if counts is None else neutralize_padding(gaussians, counts)

    def predicted_cameras(self, model_output: EncoderOutput) -> Cameras:
        pose = model_output.pred_context_pose
        h, w = model_output.depth_dict["depth"].shape[2:4]
        return Cameras(
            pose["extrinsic"].float(),
            pose["intrinsic"].float(),
            (h, w),
            normalized_intrinsics=True,
        )

    def render_planes(self, frame: ModelFrame | None) -> tuple[float, float]:
        cfg = self.config
        if frame is None:
            return cfg.znear, cfg.zfar
        return cfg.znear / frame.scale, cfg.zfar / frame.scale

    @property
    def background(self) -> tuple[float, float, float]:
        return tuple(self.config.background)

    @property
    def render_settings(self) -> dict[str, Any]:
        cfg = self.config
        return {
            "radius_clip": cfg.radius_clip,
            "rasterize_mode": cfg.rasterize_mode,
            "clamp": cfg.clamp,
        }

    # --- fitting inputs ---------------------------------------------------------

    def fit_views(self, views: Views, image_shape: list[int] | None = None) -> Views:
        """`process_image` for `crop_mode="crop"`, the ffgs rules otherwise; the
        images are quantised to 8 bit first (upstream reads 8-bit files)."""
        cfg = self.config
        shape = tuple(image_shape or cfg.image_shape)
        scaled = upstream_scaled_shape(views.image_shape, shape)
        k = None if views.intrinsics is None else views.normalized_k
        images, ks = [], []
        for i, img in enumerate(views.images):
            ki = torch.eye(3).expand(len(img), 3, 3) if k is None else k[i]
            img = quantize(img)
            if cfg.crop_mode == "crop":
                img, ki = center_crop(self._resize(img, scaled), ki, shape)
            else:
                img, ki = fit_images(
                    img,
                    ki,
                    shape,
                    crop_mode=cfg.crop_mode,
                    resize=self._resize,
                    pad_value=cfg.pad_value,
                )
            images.append(img)
            ks.append(ki)
        return Views(
            torch.stack(images),
            None if k is None else torch.stack(ks),
            views.c2w,
            normalized_intrinsics=True,
        )

    def fit_cameras(
        self, cameras: Cameras, image_shape: list[int] | None = None
    ) -> Cameras:
        cfg = self.config
        if cfg.crop_mode != "crop":
            return super().fit_cameras(cameras, image_shape)
        shape = tuple(image_shape or cfg.image_shape)
        h, w = upstream_scaled_shape(cameras.image_shape, shape)
        k = cameras.normalized_k.clone()
        k[..., 0, 0] *= w / shape[1]
        k[..., 1, 1] *= h / shape[0]
        return Cameras(cameras.c2w, k, shape, normalized_intrinsics=True)

    def _resize(self, images: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        if self.resize is not None:
            return self.resize(images, shape)
        if self.config.resize_mode == "bicubic":
            return pil_bicubic(images, shape)
        return rescale(images, shape, self.config.resize_mode)


def upstream_scaled_shape(
    image_shape: tuple[int, int], shape: tuple[int, int]
) -> tuple[int, int]:
    """The size `process_image` resizes an `image_shape` image to before cropping
    it to `shape`: the covering side exactly, the other by `int` (truncation)."""
    h, w = image_shape
    h_out, w_out = shape
    if h_out / h > w_out / w:  # upstream (square shape): width > height
        return h_out, int(w * (h_out / h))
    return int(h * (w_out / w)), w_out


def quantize(images: torch.Tensor) -> torch.Tensor:
    """[0, 1] images -> the nearest 8-bit values, as `ToTensor` gives them."""
    return (images * 255).round().clamp(0, 255).to(torch.uint8).float() / 255


def pil_bicubic(images: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """[..., 3, h, w] -> [..., 3, *shape]: PIL's bicubic resize (`Image.resize`'s
    default) of the 8-bit images, then `ToTensor`'s / 255."""
    *batch, c, h, w = images.shape
    device, dtype = images.device, images.dtype
    h_out, w_out = shape
    arrays = (images.reshape(-1, c, h, w) * 255).round().clamp(0, 255)
    arrays = arrays.to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
    resized = [
        np.asarray(Image.fromarray(a).resize((w_out, h_out), Image.BICUBIC))
        for a in arrays
    ]
    out = torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2).float() / 255
    return out.to(device=device, dtype=dtype).reshape(*batch, c, h_out, w_out)


def neutralize_padding(gaussians: Gaussians, counts: torch.Tensor) -> Gaussians:
    """Make the padding of a batch inert: Gaussians past each sample's count.

    Upstream pads the samples of a batch to the largest Gaussian count with
    placeholder values (means -1e4, scales 0, colours -1e10, opacity 0) that a
    ply cannot hold (log 0). Each placeholder becomes a transparent copy of its
    sample's first Gaussian: opacity 0, colour 0, same mean, scale and rotation.
    It renders nothing, stays within the sample's extent and is left out of
    plys. Samples without padding (any batch of one) are returned unchanged.
    """
    n = gaussians.num_gaussians
    counts = counts.to(gaussians.means.device)
    if bool((counts >= n).all()):
        return gaussians
    valid = torch.arange(n, device=counts.device)[None] < counts[:, None]  # [B, N]

    def first(x: torch.Tensor) -> torch.Tensor:
        mask = valid.view(*valid.shape, *([1] * (x.dim() - 2)))
        return torch.where(mask, x, x[:, :1])

    return Gaussians(
        means=first(gaussians.means),
        scales=first(gaussians.scales),
        quats=first(gaussians.quats),
        opacities=torch.where(valid, gaussians.opacities, 0.0),
        colors=torch.where(
            valid.view(*valid.shape, *([1] * (gaussians.colors.dim() - 2))),
            gaussians.colors,
            0.0,
        ),
        sh_degree=gaussians.sh_degree,
    )
