import math

import pytest
import torch

from ffgs import Cameras, Gaussians, ModelFrame
from ffgs.geometry import (
    matrix_to_quat,
    quat_multiply,
    quat_to_matrix,
    rotate_sh,
    sh_rotation_blocks,
)
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


def sh_basis(degree: int, dirs: torch.Tensor) -> torch.Tensor:
    """Real SH of the 3DGS / gsplat basis (Condon-Shortley phase), any degree, at
    unit `dirs` [.., 3] -> [.., (degree + 1) ** 2], from associated Legendre
    recursions: Y_lm ~ P_l^|m|(z) / sin^|m| * Re / Im (x + iy)^|m|."""
    x, y, z = dirs.unbind(-1)
    basis = {}
    for m in range(degree + 1):
        # Q_l = P_l^m(z) / sin^m(theta), P with the Condon-Shortley phase.
        q = {m: torch.full_like(z, (-1) ** m * math.prod(range(2 * m - 1, 0, -2)))}
        if m + 1 <= degree:
            q[m + 1] = z * (2 * m + 1) * q[m]
        for deg in range(m + 2, degree + 1):
            q[deg] = ((2 * deg - 1) * z * q[deg - 1] - (deg + m - 1) * q[deg - 2]) / (
                deg - m
            )
        power = torch.complex(x, y) ** m
        for deg, ql in q.items():
            norm = math.sqrt(
                (2 * deg + 1)
                / (4 * math.pi)
                * math.factorial(deg - m)
                / math.factorial(deg + m)
            )
            if m == 0:
                basis[deg, 0] = norm * ql
            else:
                basis[deg, m] = math.sqrt(2) * norm * ql * power.real
                basis[deg, -m] = math.sqrt(2) * norm * ql * power.imag
    keys = [(deg, m) for deg in range(degree + 1) for m in range(-deg, deg + 1)]
    return torch.stack([basis[key] for key in keys], dim=-1)


def eval_sh(colors: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Colour of SH coefficients [.., K, 3] along unit `dirs` [.., 3]."""
    degree = math.isqrt(colors.shape[-2]) - 1
    return (sh_basis(degree, dirs)[..., None] * colors).sum(dim=-2)


def gsplat_sh_basis(basis_dim: int, dirs: torch.Tensor) -> torch.Tensor:
    """gsplat 1.5.3 `gsplat.cuda._torch_impl._eval_sh_bases_fast` (Apache-2.0),
    verbatim but for formatting: the basis gsplat renders with, degrees <= 4."""
    result = torch.empty(
        (*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device
    )
    result[..., 0] = 0.2820947917738781
    if basis_dim <= 1:
        return result
    x, y, z = dirs.unbind(-1)
    fTmpA = -0.48860251190292
    result[..., 2] = -fTmpA * z
    result[..., 3] = fTmpA * x
    result[..., 1] = fTmpA * y
    if basis_dim <= 4:
        return result
    z2 = z * z
    fTmpB = -1.092548430592079 * z
    fTmpA = 0.5462742152960395
    fC1 = x * x - y * y
    fS1 = 2 * x * y
    result[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    result[..., 7] = fTmpB * x
    result[..., 5] = fTmpB * y
    result[..., 8] = fTmpA * fC1
    result[..., 4] = fTmpA * fS1
    if basis_dim <= 9:
        return result
    fTmpC = -2.285228997322329 * z2 + 0.4570457994644658
    fTmpB = 1.445305721320277 * z
    fTmpA = -0.5900435899266435
    fC2 = x * fC1 - y * fS1
    fS2 = x * fS1 + y * fC1
    result[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    result[..., 13] = fTmpC * x
    result[..., 11] = fTmpC * y
    result[..., 14] = fTmpB * fC1
    result[..., 10] = fTmpB * fS1
    result[..., 15] = fTmpA * fC2
    result[..., 9] = fTmpA * fS2
    if basis_dim <= 16:
        return result
    fTmpD = z * (-4.683325804901025 * z2 + 2.007139630671868)
    fTmpC = 3.31161143515146 * z2 - 0.47308734787878
    fTmpB = -1.770130769779931 * z
    fTmpA = 0.6258357354491763
    fC3 = x * fC2 - y * fS2
    fS3 = x * fS2 + y * fC2
    result[..., 20] = 1.984313483298443 * z2 * (
        1.865881662950577 * z2 - 1.119528997770346
    ) + -1.006230589874905 * (0.9461746957575601 * z2 - 0.3153915652525201)
    result[..., 21] = fTmpD * x
    result[..., 19] = fTmpD * y
    result[..., 22] = fTmpC * fC1
    result[..., 18] = fTmpC * fS1
    result[..., 23] = fTmpB * fC2
    result[..., 17] = fTmpB * fS2
    result[..., 24] = fTmpA * fC3
    result[..., 16] = fTmpA * fS3
    return result


def _random_dirs(g: torch.Generator, *shape: int) -> torch.Tensor:
    dirs = torch.randn(*shape, 3, generator=g, dtype=torch.float64)
    return torch.nn.functional.normalize(dirs, dim=-1)


def _rotate_sh1_before(colors: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """`rotate_sh` as it was for degree 1, before any degree was supported."""
    c1, c2, c3 = colors[:, :, 1], colors[:, :, 2], colors[:, :, 3]
    a = torch.stack((-c3, -c1, c2), dim=2)
    a = torch.einsum("bij,bnjc->bnic", rotation, a)
    rotated = colors.clone()
    rotated[:, :, 1] = -a[:, :, 1]
    rotated[:, :, 2] = a[:, :, 2]
    rotated[:, :, 3] = -a[:, :, 0]
    return rotated


@pytest.mark.parametrize("sh_degree", [None, 0, 1, 3])
def test_model_to_world_is_the_inverse_similarity(sh_degree) -> None:
    g = torch.Generator().manual_seed(1)
    b, n, scale = 2, 50, 0.15
    frame = ModelFrame(anchor_c2w=_random_c2w(g, b), scale=scale)
    model = random_gaussians(g, b, n, sh_degree=sh_degree)
    world = frame.model_to_world().apply(model)
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
    torch.testing.assert_close(world.opacities, model.opacities)

    # View-dependent colour: world colour along R d == model colour along d.
    if sh_degree is None:
        assert torch.equal(world.colors, model.colors)
        return
    dirs = _random_dirs(g, b, n)
    world_dirs = dirs @ rot.double().transpose(-1, -2)
    torch.testing.assert_close(
        eval_sh(world.colors.double(), world_dirs),
        eval_sh(model.colors.double(), dirs),
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.equal(world.colors[:, :, 0], model.colors[:, :, 0])


def test_similarity_keeps_the_dtype() -> None:
    g = torch.Generator().manual_seed(4)
    frame = ModelFrame(anchor_c2w=_random_c2w(g, 1), scale=0.5)
    model = random_gaussians(g, 1, 8, sh_degree=2).to(dtype=torch.float64)
    world = frame.model_to_world().apply(model)
    assert world.means.dtype == world.colors.dtype == torch.float64


@pytest.mark.parametrize(
    "factors, match",
    [
        ((1.0, 1.0, 2.0), "not orthonormal"),  # non-uniform scale
        ((2.0, 2.0, 2.0), "not orthonormal"),  # uniform, but belongs in `scale`
        ((1.0, 1.0, -1.0), "det < 0"),  # reflection
    ],
)
def test_similarity_rejects_what_is_not_one(factors, match) -> None:
    g = torch.Generator().manual_seed(6)
    c2w = _random_c2w(g, 2)
    c2w[1, :3, :3] = c2w[1, :3, :3] @ torch.diag(torch.tensor(factors))
    frame = ModelFrame(anchor_c2w=c2w, scale=0.5)
    with pytest.raises(ValueError, match=match):
        frame.model_to_world()
    with pytest.raises(ValueError, match="scale"):
        ModelFrame(anchor_c2w=c2w[:1], scale=-1.0).model_to_world()


def test_similarity_rejects_a_nonfinite_translation() -> None:
    g = torch.Generator().manual_seed(6)
    c2w = _random_c2w(g, 1)
    c2w[0, 0, 3] = float("nan")
    with pytest.raises(ValueError, match="translation must be finite"):
        ModelFrame(anchor_c2w=c2w, scale=0.5).model_to_world()


def test_frame_conversion_checks_the_batch() -> None:
    g = torch.Generator().manual_seed(7)
    frame = ModelFrame(anchor_c2w=_random_c2w(g, 2), scale=0.5)
    c2w = _random_c2w(g, 3)[None]
    cameras = Cameras(c2w, torch.eye(3).repeat(1, 3, 1, 1), (8, 8))
    with pytest.raises(ValueError, match="1 camera samples for a frame of 2"):
        frame.cameras_to_model(cameras)
    with pytest.raises(ValueError, match="1 camera samples for a frame of 2"):
        frame.c2w_to_model(c2w)
    single = ModelFrame(anchor_c2w=_random_c2w(g, 1), scale=0.5)
    with pytest.raises(ValueError, match="2 camera samples for a frame of 1"):
        single.c2w_to_model(c2w.repeat(2, 1, 1, 1))


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


def test_sh_basis_is_gsplats() -> None:
    dirs = _random_dirs(torch.Generator().manual_seed(8), 500)
    torch.testing.assert_close(
        sh_basis(4, dirs), gsplat_sh_basis(25, dirs), atol=1e-12, rtol=0
    )


@pytest.mark.parametrize("degree", range(9))
def test_rotate_sh_rotates_the_field(degree) -> None:
    # Rotated coefficients along d == original coefficients along R^T d.
    g = torch.Generator().manual_seed(10 + degree)
    b, n = 4, 32
    rotation = quat_to_matrix(torch.randn(b, 4, generator=g, dtype=torch.float64))
    colors = torch.randn(b, n, (degree + 1) ** 2, 3, generator=g, dtype=torch.float64)
    dirs = _random_dirs(g, b, n)
    rotated = rotate_sh(colors, rotation, degree)
    back = dirs @ rotation  # R^T d, per row

    def eval_gsplat(c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return (gsplat_sh_basis(c.shape[-2], d)[..., None] * c).sum(dim=-2)

    for evaluate in (eval_sh, eval_gsplat) if degree <= 4 else (eval_sh,):
        error = (evaluate(rotated, dirs) - evaluate(colors, back)).abs().max()
        assert error <= 1e-10, (evaluate.__name__, error)


def test_sh_rotation_blocks_are_orthogonal_and_compose() -> None:
    g = torch.Generator().manual_seed(20)
    r = quat_to_matrix(torch.randn(2, 6, 4, generator=g, dtype=torch.float64))
    r_a, r_b = r[0], r[1]
    blocks_a = sh_rotation_blocks(r_a, 8)
    blocks_b = sh_rotation_blocks(r_b, 8)
    blocks_ab = sh_rotation_blocks(r_a @ r_b, 8)
    for degree, (d_a, d_b, d_ab) in enumerate(zip(blocks_a, blocks_b, blocks_ab)):
        eye = torch.eye(2 * degree + 1, dtype=torch.float64)
        torch.testing.assert_close(
            d_a @ d_a.transpose(-1, -2), eye.expand_as(d_a), atol=1e-10, rtol=0
        )
        torch.testing.assert_close(d_a @ d_b, d_ab, atol=1e-10, rtol=0)

    # Norm per degree is kept.
    colors = torch.randn(6, 10, 81, 3, generator=g, dtype=torch.float64)
    rotated = rotate_sh(colors, r_a, 8)
    for degree in range(9):
        band = slice(degree**2, (degree + 1) ** 2)
        torch.testing.assert_close(
            rotated[:, :, band].norm(dim=2),
            colors[:, :, band].norm(dim=2),
            atol=1e-10,
            rtol=0,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotate_sh_identity_and_degree_one(dtype) -> None:
    g = torch.Generator().manual_seed(30)
    colors = torch.randn(3, 20, 25, 3, generator=g).to(dtype)
    eye = torch.eye(3, dtype=dtype).repeat(3, 1, 1)
    assert torch.equal(rotate_sh(colors, eye, 4), colors)

    # Degree 1: bit for bit what it was, and the recursion's D_1 in value.
    rotation = quat_to_matrix(torch.randn(3, 4, generator=g)).to(dtype)
    first = colors[:, :, :4]
    rotated = rotate_sh(first, rotation, 1)
    assert torch.equal(rotated, _rotate_sh1_before(first, rotation))
    d1 = sh_rotation_blocks(rotation.double(), 1)[1].to(dtype)
    torch.testing.assert_close(
        rotated[:, :, 1:], torch.einsum("bij,bnjc->bnic", d1, first[:, :, 1:])
    )
    # The degree-1 band of a degree-4 field rotates the same way.
    assert torch.equal(rotate_sh(colors, rotation, 4)[:, :, :4], rotated)


def test_rotate_sh_checks_the_coefficient_count() -> None:
    with pytest.raises(ValueError, match="needs 9"):
        rotate_sh(torch.zeros(1, 2, 4, 3), torch.eye(3)[None], 2)


def test_sh_degree_from_channels() -> None:
    assert sh_degree_from_channels(3) is None
    assert sh_degree_from_channels(12) == 1
    assert sh_degree_from_channels(48) == 3
    with pytest.raises(ValueError):
        sh_degree_from_channels(14)
