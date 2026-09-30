"""Rasterise `Gaussians` at `Cameras` with gsplat (imported on first use)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .types import Cameras, Gaussians


def _rasterization():
    try:
        from gsplat import rasterization
    except ImportError:
        try:
            from gsplat.rendering import rasterization
        except ImportError as err:
            raise ImportError(
                'rendering needs gsplat: pip install "ffgs[render]"'
            ) from err
    return rasterization


def render(
    gaussians: Gaussians,
    cameras: Cameras,
    *,
    near: float,
    far: float,
    background: Sequence[float] | None = (0.5, 0.5, 0.5),
    render_depth: bool = False,
) -> dict[str, torch.Tensor]:
    """-> {"images": [B, V, 3, H, W], "alphas": [B, V, 1, H, W], ["depths"]}.

    `near` / `far` are in the units of the frame the Gaussians and cameras share.
    """
    rasterization = _rasterization()
    h, w = cameras.image_shape
    device = gaussians.means.device
    viewmats = torch.inverse(cameras.c2w.float()).to(device)
    ks = cameras.pixel_k.to(device)
    b, v = viewmats.shape[:2]

    means = gaussians.means.contiguous().float()
    quats = gaussians.quats.contiguous().float()
    scales = gaussians.scales.contiguous().float()
    opacities = gaussians.opacities.contiguous().float()
    colors = gaussians.colors.contiguous().float()
    kwargs = {}
    if gaussians.sh_degree is not None:
        kwargs["sh_degree"] = gaussians.sh_degree
    backgrounds = None
    if background is not None:
        backgrounds = torch.tensor(background, dtype=torch.float32, device=device)
        backgrounds = backgrounds[None].repeat(v, 1)

    images, alphas, depths = [], [], []
    for i in range(b):
        rendered, alpha, _ = rasterization(
            means=means[i],
            quats=quats[i],
            scales=scales[i],
            opacities=opacities[i],
            colors=colors[i],
            viewmats=viewmats[i],
            Ks=ks[i],
            width=w,
            height=h,
            near_plane=near,
            far_plane=far,
            packed=False,
            backgrounds=backgrounds,
            render_mode="RGB+ED" if render_depth else "RGB",
            **kwargs,
        )
        images.append(rendered[..., :3].permute(0, 3, 1, 2))
        alphas.append(alpha.permute(0, 3, 1, 2))
        if render_depth:
            depths.append(rendered[..., 3:].permute(0, 3, 1, 2))

    out = {"images": torch.stack(images), "alphas": torch.stack(alphas)}
    if render_depth:
        out["depths"] = torch.stack(depths)
    return out
