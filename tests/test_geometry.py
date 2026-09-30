import pytest
import torch

from ffgs import Gaussians, ModelFrame
from ffgs.geometry import matrix_to_quat, quat_multiply, quat_to_matrix, rotate_sh
from ffgs.types import sh_degree_from_channels


def _random_rotation(generator: torch.Generator, n: int) -> torch.Tensor:
    return quat_to_matrix(torch.randn(n, 4, generator=generator))


def _random_c2w(generator: torch.Generator, n: int) -> torch.Tensor:
    c2w = torch.eye(4).repeat(n, 1, 1)
    c2w[:, :3, :3] = _random_rotation(generator, n)
    c2w[:, :3, 3] = torch.randn(n, 3, generator=generator) * 3
    return c2w


def random_gaussians(g: torch.Generator, b: int, n: int, sh_degree) -> Gaussians:
    colors = (
        torch.rand(b, n, 3, generator=g)
        if sh_degree is None
        else torch.randn(b, n, (sh_degree + 1) ** 2, 3, generator=g)
    )
    return Gaussians(
        means=torch.randn(b, n, 3, generator=g, dtype=torch.float64).float(),
        scales=torch.rand(b, n, 3, generator=g) * 0.1 + 0.01,
        quats=torch.nn.functional.normalize(torch.randn(b, n, 4, generator=g), dim=-1),
        opacities=torch.rand(b, n, generator=g),
        colors=colors,
        sh_degree=sh_degree,
    )


def _eval_sh1(colors: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """gsplat / 3DGS degree <= 1 SH colour along unit `dirs` [.., 3]."""
    x, y, z = dirs.unbind(-1)
    c0 = 0.28209479177387814
    c1 = 0.4886025119029199
    out = c0 * colors[..., 0, :]
    return out + c1 * (
        -y[..., None] * colors[..., 1, :]
        + z[..., None] * colors[..., 2, :]
        - x[..., None] * colors[..., 3, :]
    )


def test_gaussians_to_world_is_the_inverse_similarity() -> None:
    g = torch.Generator().manual_seed(1)
    b, n, scale = 2, 50, 0.15
    frame = ModelFrame(anchor_c2w=_random_c2w(g, b), scale=scale)
    model = random_gaussians(g, b, n, sh_degree=1)
    world = frame.gaussians_to_world(model)
    rot = frame.anchor_c2w[:, :3, :3]

    # Means: world -> model with the camera arithmetic recovers the input.
    homog = torch.cat([world.means, torch.ones(b, n, 1)], dim=-1)
    back = (homog @ torch.inverse(frame.anchor_c2w).transpose(-1, -2))[..., :3]
    torch.testing.assert_close(back * scale, model.means, atol=1e-5, rtol=1e-5)

    # Covariances: Sigma_w = R Sigma_m R^T / s^2.
    def cov(gs: Gaussians) -> torch.Tensor:
        r = quat_to_matrix(gs.quats)
        s = torch.diag_embed(gs.scales**2)
        return r @ s @ r.transpose(-1, -2)

    want = rot[:, None] @ cov(model) @ rot[:, None].transpose(-1, -2) / scale**2
    torch.testing.assert_close(cov(world), want, atol=1e-5, rtol=1e-4)

    # View-dependent colour: world colour along R d == model colour along d.
    dirs = torch.nn.functional.normalize(torch.randn(b, n, 3, generator=g), dim=-1)
    world_dirs = dirs @ rot.transpose(-1, -2)
    torch.testing.assert_close(
        _eval_sh1(world.colors, world_dirs),
        _eval_sh1(model.colors, dirs),
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(world.opacities, model.opacities)


def test_c2w_to_model_anchors_the_first_view() -> None:
    g = torch.Generator().manual_seed(5)
    c2w = _random_c2w(g, 4)[None]
    frame = ModelFrame(anchor_c2w=c2w[:, 0], scale=0.5)
    model = frame.c2w_to_model(c2w)
    torch.testing.assert_close(model[0, 0], torch.eye(4), atol=1e-6, rtol=0)
    # Relative rotations are kept; relative translations are scaled.
    rel = torch.inverse(c2w[0, 0]) @ c2w[0, 1]
    torch.testing.assert_close(model[0, 1, :3, :3], rel[:3, :3], atol=1e-5, rtol=0)
    torch.testing.assert_close(model[0, 1, :3, 3], rel[:3, 3] * 0.5, atol=1e-5, rtol=0)


def test_rgb_gaussians_keep_colors() -> None:
    g = torch.Generator().manual_seed(2)
    frame = ModelFrame(anchor_c2w=_random_c2w(g, 1), scale=0.25)
    model = random_gaussians(g, 1, 10, sh_degree=None)
    assert torch.equal(frame.gaussians_to_world(model).colors, model.colors)


def test_quaternion_helpers() -> None:
    g = torch.Generator().manual_seed(3)
    r = _random_rotation(g, 100)
    torch.testing.assert_close(quat_to_matrix(matrix_to_quat(r)), r, atol=1e-5, rtol=0)
    a, b = matrix_to_quat(r[:50]), matrix_to_quat(r[50:])
    torch.testing.assert_close(
        quat_to_matrix(quat_multiply(a, b)), r[:50] @ r[50:], atol=1e-5, rtol=0
    )
    # 180-degree rotations, where the trace-only formula breaks down.
    flips = torch.diag_embed(torch.tensor([[1.0, -1, -1], [-1, 1, -1], [-1, -1, 1]]))
    torch.testing.assert_close(quat_to_matrix(matrix_to_quat(flips)), flips)


def test_rotate_sh_above_degree_one_is_not_implemented() -> None:
    colors = torch.zeros(1, 2, 9, 3)
    with pytest.raises(NotImplementedError):
        rotate_sh(colors, torch.eye(3)[None], 2)


def test_sh_degree_from_channels() -> None:
    assert sh_degree_from_channels(3) is None
    assert sh_degree_from_channels(12) == 1
    assert sh_degree_from_channels(48) == 3
    with pytest.raises(ValueError):
        sh_degree_from_channels(14)
