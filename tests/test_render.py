import importlib
import importlib.util

import pytest
import torch
from test_geometry import random_gaussians

from ffgs import Cameras, render

# The module (`ffgs.render` is the function).
render_module = importlib.import_module("ffgs.render")

HAS_GSPLAT = importlib.util.find_spec("gsplat") is not None


@pytest.mark.skipif(HAS_GSPLAT, reason="gsplat is installed")
def test_render_without_gsplat_says_how_to_install() -> None:
    gs = random_gaussians(torch.Generator().manual_seed(0), 1, 4, sh_degree=None)
    cams = Cameras(torch.eye(4)[None], torch.eye(3)[None], (8, 8), True)
    with pytest.raises(ImportError, match=r"ffgs\[render\]"):
        render(gs, cams, near=0.01, far=100.0)


@pytest.mark.gpu
@pytest.mark.parametrize("sh_degree", [None, 1])
def test_render_shapes(sh_degree) -> None:
    g = torch.Generator().manual_seed(0)
    gs = random_gaussians(g, 2, 64, sh_degree=sh_degree).to("cuda")
    gs.means[..., 2] += 4.0  # in front of the cameras
    k = torch.tensor([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]])
    cams = Cameras(
        torch.eye(4).repeat(2, 3, 1, 1), k.repeat(2, 3, 1, 1), (24, 32), True
    )
    out = render(gs, cams.to("cuda"), near=0.01, far=100.0, render_depth=True)
    assert out["images"].shape == (2, 3, 3, 24, 32)
    assert out["alphas"].shape == (2, 3, 1, 24, 32)
    assert out["depths"].shape == (2, 3, 1, 24, 32)
    assert out["alphas"].max() > 0


def test_render_passes_the_settings_to_gsplat(monkeypatch) -> None:
    calls = []

    def rasterization(**kwargs):
        calls.append(kwargs)
        v, h, w = len(kwargs["viewmats"]), kwargs["height"], kwargs["width"]
        rendered = torch.linspace(-1, 2, v * h * w * 3).reshape(v, h, w, 3)
        return rendered, torch.ones(v, h, w, 1), {}

    monkeypatch.setattr(render_module, "_rasterization", lambda: rasterization)
    gs = random_gaussians(torch.Generator().manual_seed(0), 1, 4, sh_degree=None)
    cams = Cameras(torch.eye(4)[None], torch.eye(3)[None], (8, 8), True)
    out = render(gs, cams, near=0.01, far=100.0)
    assert calls[-1]["radius_clip"] == 0.0
    assert calls[-1]["rasterize_mode"] == "classic"
    assert out["images"].min() == -1 and out["images"].max() == 2
    out = render(
        gs,
        cams,
        near=0.01,
        far=100.0,
        radius_clip=0.1,
        rasterize_mode="antialiased",
        clamp=True,
    )
    assert calls[-1]["radius_clip"] == 0.1
    assert calls[-1]["rasterize_mode"] == "antialiased"
    assert out["images"].min() == 0 and out["images"].max() == 1


def _cameras(b: int, v: int, shape=(24, 32)) -> Cameras:
    """Cameras along x, all facing +z, each with its own focal length."""
    c2w = torch.eye(4).repeat(b, v, 1, 1)
    c2w[..., 0, 3] = torch.linspace(-0.5, 0.5, b * v).reshape(b, v)
    k = torch.tensor([[1.0, 0, 0.5], [0, 1.3, 0.5], [0, 0, 1]]).repeat(b, v, 1, 1)
    k[..., 0, 0] += torch.linspace(0.0, 0.4, b * v).reshape(b, v)
    return Cameras(c2w, k, shape, True)


def test_render_rasterises_one_camera_per_call(monkeypatch) -> None:
    calls = []

    def rasterization(**kwargs):
        calls.append(kwargs)
        (c,), h, w = kwargs["viewmats"].shape[:1], kwargs["height"], kwargs["width"]
        # Each camera's image is filled with its viewmat x translation.
        value = kwargs["viewmats"][:, 0, 3].reshape(c, 1, 1, 1)
        return value.expand(c, h, w, 4).clone(), value.expand(c, h, w, 1), {}

    monkeypatch.setattr(render_module, "_rasterization", lambda: rasterization)
    gs = random_gaussians(torch.Generator().manual_seed(0), 2, 4, sh_degree=3)
    cams = _cameras(2, 3)
    out = render(gs, cams, near=0.01, far=100.0, render_depth=True)

    assert len(calls) == 6
    assert all(len(c["viewmats"]) == 1 and len(c["Ks"]) == 1 for c in calls)
    assert all(c["backgrounds"].shape == (1, 3) for c in calls)
    assert all(c["sh_degree"] == 3 for c in calls)
    # Images in camera order, per sample: x of the viewmat = -x of the c2w.
    want = -cams.c2w[..., 0, 3]
    assert torch.equal(out["images"][:, :, 0, 0, 0], want)
    assert torch.equal(out["alphas"][:, :, 0, 0, 0], want)
    assert torch.equal(out["depths"][:, :, 0, 0, 0], want)
    assert out["images"].shape == (2, 3, 3, 24, 32)


@pytest.mark.gpu
def test_render_per_camera_equals_one_call_for_all_cameras() -> None:
    """The per-camera calls give gsplat's all-camera result bit for bit."""
    from gsplat import rasterization

    g = torch.Generator().manual_seed(0)
    gs = random_gaussians(g, 2, 500, sh_degree=3).to("cuda")
    gs.means[..., 2] = gs.means[..., 2].abs() + 2.0  # in front of the cameras
    cams = _cameras(2, 4).to("cuda")
    background = (0.2, 0.5, 0.8)
    out = render(
        gs, cams, near=0.01, far=100.0, background=background, render_depth=True
    )
    assert out["alphas"].max() > 0.1  # something was drawn

    backgrounds = torch.tensor(background, device="cuda").repeat(4, 1)  # [C, 3]
    for i in range(2):
        rendered, alpha, _ = rasterization(
            means=gs.means[i],
            quats=gs.quats[i],
            scales=gs.scales[i],
            opacities=gs.opacities[i],
            colors=gs.colors[i],
            viewmats=torch.inverse(cams.c2w[i]),
            Ks=cams.pixel_k[i],
            width=32,
            height=24,
            near_plane=0.01,
            far_plane=100.0,
            packed=False,
            backgrounds=backgrounds,
            render_mode="RGB+ED",
            sh_degree=3,
        )
        assert torch.equal(out["images"][i], rendered[..., :3].permute(0, 3, 1, 2))
        assert torch.equal(out["depths"][i], rendered[..., 3:].permute(0, 3, 1, 2))
        assert torch.equal(out["alphas"][i], alpha.permute(0, 3, 1, 2))
