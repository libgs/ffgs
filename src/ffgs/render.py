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
    near: float | torch.Tensor,
    far: float | torch.Tensor,
    background: Sequence[float] | None = (0.5, 0.5, 0.5),
    render_depth: bool = False,
    radius_clip: float = 0.0,
    rasterize_mode: str = "classic",
    clamp: bool = False,
) -> dict[str, torch.Tensor]:
    """-> {"images": [B, V, 3, H, W], "alphas": [B, V, 1, H, W], ["depths"]}.

    `near` / `far` are in the units of the frame the Gaussians and cameras share:
    one value, or one per sample ([B]). `radius_clip` (pixels) and
    `rasterize_mode` ("classic" / "antialiased") are gsplat's; `clamp` clips the
    images to [0, 1]. A model's own values come from its processor
    (`Processor.render_settings`, applied by `GSPipeline.render`).
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
        backgrounds = backgrounds[None]  # [1, 3]: one camera per call

    images, alphas, depths = [], [], []
    for i in range(b):
        # One camera per call: for SH colours gsplat materialises the coefficients
        # per camera ([C, N, K, 3]), so memory would grow with the number of
        # cameras. Cameras are rasterised independently; the images are the same.
        per_camera = [
            rasterization(
                means=means[i],
                quats=quats[i],
                scales=scales[i],
                opacities=opacities[i],
                colors=colors[i],
                viewmats=viewmats[i, j : j + 1],
                Ks=ks[i, j : j + 1],
                width=w,
                height=h,
                near_plane=_per_sample(near, i),
                far_plane=_per_sample(far, i),
                packed=False,
                backgrounds=backgrounds,
                render_mode="RGB+ED" if render_depth else "RGB",
                radius_clip=radius_clip,
                rasterize_mode=rasterize_mode,
                **kwargs,
            )[:2]
            for j in range(v)
        ]
        rendered = torch.cat([r for r, _ in per_camera])
        alpha = torch.cat([a for _, a in per_camera])
        image = rendered[..., :3]
        if clamp:
            image = image.clamp(0.0, 1.0)
        images.append(image.permute(0, 3, 1, 2))
        alphas.append(alpha.permute(0, 3, 1, 2))
        if render_depth:
            depths.append(rendered[..., 3:].permute(0, 3, 1, 2))

    out = {"images": torch.stack(images), "alphas": torch.stack(alphas)}
    if render_depth:
        out["depths"] = torch.stack(depths)
    return out


def _per_sample(value: float | torch.Tensor, index: int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.reshape(-1)[index] if value.numel() > 1 else value)
    return value
