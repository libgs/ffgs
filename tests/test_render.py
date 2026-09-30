import importlib.util

import pytest
import torch
from test_geometry import random_gaussians

from ffgs import Cameras, render

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
