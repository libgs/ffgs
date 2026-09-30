# Adapted from the crop_shim of pixelSplat (https://github.com/dcharatan/pixelsplat):
# Copyright (c) 2023 David Charatan, Sizhe Li, Andrea Tagliasacchi, and Vincent Sitzmann
# SPDX-License-Identifier: MIT (see the pixelSplat section at the end of LICENSE)

"""Fitting input images to a model's input shape.

The default (`crop_mode="crop"`) is the rule most feed-forward 3DGS models are
trained with: scale to cover the shape, crop the middle, keep the normalised
principal point. `tests/test_image.py` pins it bit for bit to frozen reference
outputs (`tests/fixtures/`). `"pad"` scales to fit inside and pads instead
(nothing is cut off); `"none"` takes inputs already at the model's shape as they are.

Intrinsics are the normalised 3x3 (fx / W, cx / W, fy / H, cy / H). The centre crop
scales fx / fy and leaves the normalised principal point where it is -- the rule the
models were trained with, which assumes a centred principal point. Padding moves the
principal point with the content.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

RESIZE_MODES = ("lanczos", "bilinear")
CROP_MODES = ("crop", "pad", "none")

# (images [..., 3, h, w], (h_out, w_out)) -> images [..., 3, h_out, w_out]
Resize = Callable[[torch.Tensor, tuple[int, int]], torch.Tensor]


def rescale(
    images: torch.Tensor,
    shape: tuple[int, int],
    resize_mode: str = "lanczos",
) -> torch.Tensor:
    """[..., 3, h_in, w_in] float in [0, 1] -> [..., 3, h_out, w_out]."""
    *batch, c, h, w = images.shape
    images = images.reshape(-1, c, h, w)
    h_out, w_out = shape

    if resize_mode == "bilinear":
        images_new = F.resize(
            images,
            (h_out, w_out),
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
        )
    elif resize_mode == "lanczos":
        # PIL takes uint8 only, so this path quantises to 8 bit before resampling.
        device, dtype = images.device, images.dtype
        imgs_np = (images * 255).clip(min=0, max=255).to(torch.uint8)
        imgs_np = imgs_np.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        rescaled_list = [
            np.array(Image.fromarray(img).resize((w_out, h_out), Image.LANCZOS))
            for img in imgs_np
        ]
        images_new = torch.from_numpy(np.stack(rescaled_list) / 255.0)
        images_new = images_new.to(dtype=dtype, device=device).permute(0, 3, 1, 2)
    else:
        raise ValueError(
            f"Unknown resize_mode {resize_mode!r}; expected one of {RESIZE_MODES}."
        )

    return images_new.reshape(*batch, c, h_out, w_out)


def center_crop(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    *_, h_in, w_in = images.shape
    h_out, w_out = shape

    # Odd size differences induce half-pixel misalignments, as in training.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2
    images = images[..., :, row : row + h_out, col : col + w_out]

    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy
    return images, intrinsics


def scaled_shape(
    image_shape: tuple[int, int], shape: tuple[int, int], crop_mode: str
) -> tuple[int, int]:
    """The size an image is resized to before the crop / pad to `shape`."""
    h_in, w_in = image_shape
    h_out, w_out = shape
    if crop_mode == "none":
        return h_in, w_in
    pick = max if crop_mode == "crop" else min
    scale_factor = pick(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out
    return h_scaled, w_scaled


def rescale_and_crop(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    shape: tuple[int, int],
    resize_mode: str = "lanczos",
    resize: Resize | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale so the image covers `shape`, then crop the middle.

    `resize` replaces the built-in `resize_mode` kernels when given.
    """
    scaled = scaled_shape(tuple(images.shape[-2:]), shape, "crop")
    if resize is None:
        images = rescale(images, scaled, resize_mode)
    else:
        images = resize(images, scaled)
    return center_crop(images, intrinsics, shape)


def rescale_and_pad(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    shape: tuple[int, int],
    resize_mode: str = "lanczos",
    resize: Resize | None = None,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scale so the image fits inside `shape`, then pad around it (centred)."""
    image_shape = tuple(images.shape[-2:])
    h_scaled, w_scaled = scaled_shape(image_shape, shape, "pad")
    if resize is None:
        images = rescale(images, (h_scaled, w_scaled), resize_mode)
    else:
        images = resize(images, (h_scaled, w_scaled))
    h_out, w_out = shape
    top, left = (h_out - h_scaled) // 2, (w_out - w_scaled) // 2
    padded = images.new_full((*images.shape[:-2], h_out, w_out), pad_value)
    padded[..., top : top + h_scaled, left : left + w_scaled] = images
    return padded, fit_intrinsics(intrinsics, image_shape, shape, "pad")


def fit_intrinsics(
    intrinsics: torch.Tensor,
    image_shape: tuple[int, int],
    shape: tuple[int, int],
    crop_mode: str = "crop",
) -> torch.Tensor:
    """Normalised K for an `image_shape` image after `fit_images` to `shape`."""
    check_crop_mode(crop_mode)
    h_out, w_out = shape
    if crop_mode == "none":
        check_shape(image_shape, shape)
        return intrinsics.clone()
    h_scaled, w_scaled = scaled_shape(image_shape, shape, crop_mode)
    k = intrinsics.clone()
    k[..., 0, 0] *= w_scaled / w_out
    k[..., 1, 1] *= h_scaled / h_out
    if crop_mode == "pad":
        top, left = (h_out - h_scaled) // 2, (w_out - w_scaled) // 2
        k[..., 0, 2] = (k[..., 0, 2] * w_scaled + left) / w_out
        k[..., 1, 2] = (k[..., 1, 2] * h_scaled + top) / h_out
    return k


def fit_images(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    shape: tuple[int, int],
    resize_mode: str = "lanczos",
    crop_mode: str = "crop",
    resize: Resize | None = None,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """[..., 3, h, w] images + normalised K -> `shape` by `crop_mode`."""
    check_crop_mode(crop_mode)
    shape = tuple(shape)
    if crop_mode == "crop":
        return rescale_and_crop(images, intrinsics, shape, resize_mode, resize)
    if crop_mode == "pad":
        return rescale_and_pad(
            images, intrinsics, shape, resize_mode, resize, pad_value
        )
    check_shape(tuple(images.shape[-2:]), shape)
    return images, intrinsics


def check_crop_mode(crop_mode: str) -> None:
    if crop_mode not in CROP_MODES:
        raise ValueError(
            f"Unknown crop_mode {crop_mode!r}; expected one of {CROP_MODES}."
        )


def check_shape(image_shape: tuple[int, int], shape: tuple[int, int]) -> None:
    if tuple(image_shape) != tuple(shape):
        raise ValueError(
            f'crop_mode="none" needs inputs at the model shape {tuple(shape)}, '
            f"got {tuple(image_shape)}"
        )
