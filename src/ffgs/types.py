"""The standard inputs and outputs every ffgs model speaks.

Conventions, the same for every model:

- images: float in [0, 1] (uint8 is accepted and converted), [V, 3, H, W] or
  [B, V, 3, H, W];
- c2w: camera-to-world, OpenCV axes (x right, y down, z forward), [.., 4, 4];
- intrinsics: pixel-unit K [.., 3, 3] for the image size given, or the normalised K
  (fx / W, cx / W, fy / H, cy / H) with `normalized_intrinsics=True`; zero skew
  (the renderers have no skew term, so a nonzero K[0, 1] is rejected);
- `Gaussians` come back in the model's frame; `GSOutput.to_world()` takes them
  to the world frame of the c2w they were predicted from.

Unbatched inputs get a leading batch dimension of 1; outputs stay batched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.v2.functional import to_dtype

SH_C0 = 0.28209479177387814


def _batched(x: torch.Tensor, ndim: int, name: str) -> torch.Tensor:
    if x.ndim == ndim - 1:
        return x[None]
    if x.ndim != ndim:
        raise ValueError(f"{name} must have {ndim - 1} or {ndim} dims, got {x.shape}")
    return x


def _check_finite(x: torch.Tensor, name: str) -> None:
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} has nonfinite values (NaN or Inf)")


def _check_no_skew(intrinsics: torch.Tensor) -> None:
    skew = intrinsics[..., 0, 1].abs()
    if (skew > 1e-6 * intrinsics[..., 0, 0].abs()).any():
        raise ValueError(
            "intrinsics have a nonzero skew K[0, 1], which the renderers do not "
            f"support (max |skew| {skew.max().item():.3g})"
        )


def normalize_intrinsics(
    intrinsics: torch.Tensor, image_shape: tuple[int, int]
) -> torch.Tensor:
    """Pixel K -> normalised K for an image of `image_shape` (h, w)."""
    h, w = image_shape
    k = intrinsics.float().clone()
    k[..., 0, :] /= w
    k[..., 1, :] /= h
    return k


def pixel_intrinsics(
    intrinsics: torch.Tensor, image_shape: tuple[int, int]
) -> torch.Tensor:
    """Normalised K -> pixel K [.., 3, 3] (zero skew), as the renderers build it."""
    h, w = image_shape
    k = intrinsics.new_zeros(intrinsics.shape, dtype=torch.float32)
    k[..., 0, 0] = intrinsics[..., 0, 0] * w
    k[..., 1, 1] = intrinsics[..., 1, 1] * h
    k[..., 0, 2] = intrinsics[..., 0, 2] * w
    k[..., 1, 2] = intrinsics[..., 1, 2] * h
    k[..., 2, 2] = 1.0
    return k


@dataclass
class Cameras:
    """Cameras to render at: c2w [B, V, 4, 4], K [B, V, 3, 3], image (h, w)."""

    c2w: torch.Tensor
    intrinsics: torch.Tensor
    image_shape: tuple[int, int]
    normalized_intrinsics: bool = False

    def __post_init__(self) -> None:
        self.c2w = _batched(self.c2w, 4, "c2w")
        _check_finite(self.c2w, "c2w")
        self.intrinsics = _batched(self.intrinsics, 4, "intrinsics")
        _check_finite(self.intrinsics, "intrinsics")
        _check_no_skew(self.intrinsics)
        self.image_shape = tuple(int(x) for x in self.image_shape)
        if self.c2w.shape[:2] != self.intrinsics.shape[:2]:
            raise ValueError(
                f"c2w {tuple(self.c2w.shape)} and intrinsics "
                f"{tuple(self.intrinsics.shape)} disagree on [B, V]"
            )

    @property
    def normalized_k(self) -> torch.Tensor:
        if self.normalized_intrinsics:
            return self.intrinsics.float()
        return normalize_intrinsics(self.intrinsics, self.image_shape)

    @property
    def pixel_k(self) -> torch.Tensor:
        return pixel_intrinsics(self.normalized_k, self.image_shape)

    def to(self, device: torch.device | str) -> Cameras:
        return replace(
            self, c2w=self.c2w.to(device), intrinsics=self.intrinsics.to(device)
        )


@dataclass
class Views:
    """Posed input images. See the module docstring for conventions."""

    images: torch.Tensor
    intrinsics: torch.Tensor
    c2w: torch.Tensor
    normalized_intrinsics: bool = False

    def __post_init__(self) -> None:
        images = _batched(self.images, 5, "images")
        if images.dtype == torch.uint8:
            images = to_dtype(images, torch.float32, scale=True)
        if images.shape[2] != 3:
            raise ValueError(f"images must be RGB, got {tuple(images.shape)}")
        self.images = images
        self.intrinsics = _batched(self.intrinsics, 4, "intrinsics")
        _check_finite(self.intrinsics, "intrinsics")
        _check_no_skew(self.intrinsics)
        self.c2w = _batched(self.c2w, 4, "c2w")
        _check_finite(self.c2w, "c2w")
        if not images.shape[:2] == self.intrinsics.shape[:2] == self.c2w.shape[:2]:
            raise ValueError("images, intrinsics and c2w disagree on [B, V]")

    @property
    def image_shape(self) -> tuple[int, int]:
        return tuple(self.images.shape[-2:])

    @property
    def cameras(self) -> Cameras:
        return Cameras(
            self.c2w, self.intrinsics, self.image_shape, self.normalized_intrinsics
        )

    @property
    def normalized_k(self) -> torch.Tensor:
        return self.cameras.normalized_k


@dataclass
class Gaussians:
    """3D Gaussians [B, N, ...].

    means [B, N, 3]; scales [B, N, 3] (linear); quats [B, N, 4] (wxyz, unit);
    opacities [B, N] in [0, 1] (0: invisible, such as the padding of a batch);
    colors [B, N, K, 3] SH coefficients with K = (sh_degree + 1) ** 2, or [B, N, 3]
    RGB in [0, 1] when `sh_degree` is None.
    """

    means: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor
    opacities: torch.Tensor
    colors: torch.Tensor
    sh_degree: int | None

    def __post_init__(self) -> None:
        if self.sh_degree is None:
            if self.colors.shape[-1] != 3 or self.colors.ndim != 3:
                raise ValueError("RGB colors must be [B, N, 3]")
        elif self.colors.shape[-2:] != ((self.sh_degree + 1) ** 2, 3):
            raise ValueError(
                f"SH degree {self.sh_degree} colors must be [B, N, "
                f"{(self.sh_degree + 1) ** 2}, 3], got {tuple(self.colors.shape)}"
            )

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[1]

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> Gaussians:
        def move(x: torch.Tensor) -> torch.Tensor:
            return x.to(device=device, dtype=dtype)

        return replace(
            self,
            means=move(self.means),
            scales=move(self.scales),
            quats=move(self.quats),
            opacities=move(self.opacities),
            colors=move(self.colors),
        )

    def sh(self) -> torch.Tensor:
        """Colors as SH coefficients [B, N, K, 3] (RGB -> degree 0)."""
        if self.sh_degree is not None:
            return self.colors
        return ((self.colors - 0.5) / SH_C0)[..., None, :]

    def save_ply(self, path: str | Path, index: int = 0, *, prune: bool = True) -> Path:
        """Write sample `index` as a standard 3DGS ply (INRIA attribute layout).

        Binary little-endian float32 per attribute, written directly: the format is
        a text header plus one record array, no library needed. `prune` leaves out
        the Gaussians with opacity exactly 0 (such as the padding of a batch),
        which render nothing; `prune=False` writes every Gaussian, opacity clamped
        to [1e-6, 1 - 1e-6] for the logit.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keep = self.opacities[index].detach().cpu() != 0
        if not prune:
            keep = torch.ones_like(keep)

        def sample(x: torch.Tensor) -> torch.Tensor:
            return x[index].detach().float().cpu()[keep]

        sh = sample(self.sh())
        means = sample(self.means).numpy()
        f_dc = sh[:, 0, :].numpy()
        # Channel-major rest coefficients: R_1..R_k, G_1..G_k, B_1..B_k.
        f_rest = sh[:, 1:, :].transpose(1, 2).flatten(1).numpy()
        opacity = torch.logit(sample(self.opacities).clamp(1e-6, 1 - 1e-6))
        opacity = opacity[:, None].numpy()
        scale = torch.log(sample(self.scales)).numpy()
        rot = sample(self.quats).numpy()

        names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        names += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
        names += ["opacity", "scale_0", "scale_1", "scale_2"]
        names += ["rot_0", "rot_1", "rot_2", "rot_3"]
        data = np.concatenate(
            [means, np.zeros_like(means), f_dc, f_rest, opacity, scale, rot], axis=1
        )
        header = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {data.shape[0]}",
            *(f"property float {name}" for name in names),
            "end_header",
        ]
        with path.open("wb") as handle:
            handle.write(("\n".join(header) + "\n").encode("ascii"))
            handle.write(np.ascontiguousarray(data, dtype="<f4").tobytes())
        return path


def sh_degree_from_channels(channels: int) -> int | None:
    """Flattened color channels -> SH degree (None for plain RGB)."""
    if channels == 3:
        return None
    k, rem = divmod(channels, 3)
    side = math.isqrt(k)
    if rem or side * side != k:
        raise ValueError(f"{channels} color channels are neither RGB nor SH")
    return side - 1
