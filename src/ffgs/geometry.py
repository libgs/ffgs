"""The model's working frame, and moving cameras / Gaussians in and out of it.

Feed-forward models are trained on normalised cameras: here, every pose relative to
the first context view, then translations scaled by a constant `scene_scale`. A
`ModelFrame` records that normalisation per sample, so inputs go in with exactly the
training arithmetic, and predicted Gaussians can be taken to the caller's world
frame by one `Similarity` shared by every model (SH rotated to any degree).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch

from .types import Cameras, Gaussians

# Tolerance of R R^T = I for a rotation taken from caller c2w (fp32, maybe parsed
# from text): loose enough for rounded poses, tight enough to catch any scale.
_ORTHONORMAL_ATOL = 1e-3


def _check_rotation(r: torch.Tensor, what: str) -> None:
    """Raise unless every r [.., 3, 3] is a proper rotation."""
    r = r.double()
    if not torch.isfinite(r).all():
        raise ValueError(f"{what}: the rotation is not finite")
    eye = torch.eye(3, dtype=r.dtype, device=r.device)
    error = (r @ r.transpose(-1, -2) - eye).abs().amax(dim=(-2, -1))
    if (error > _ORTHONORMAL_ATOL).any():
        raise ValueError(
            f"{what}: the rotation is not orthonormal (non-uniform scale or "
            f"shear; max |R R^T - I| = {error.max().item():.2e})"
        )
    if (torch.linalg.det(r) < 0).any():
        raise ValueError(
            f"{what}: the rotation has det < 0 (a reflection, e.g. left-handed "
            "c2w axes)"
        )


@dataclass
class PoseResiduals:
    """How far the cameras a fitted frame maps to the world land from the given
    ones, per view [B, V]."""

    rotation_deg: torch.Tensor  # angle between the mapped and the given rotation
    center_error: torch.Tensor  # distance between the camera centres, world units
    # center_error over the RMS distance of the given centres from their mean.
    relative_center_error: torch.Tensor


@dataclass
class ModelFrame:
    """world -> model: x_m = scale * inverse(anchor_c2w) @ x_w, per sample.

    `scale` is one number for the batch, or a [B] tensor (fitted frames).
    `residuals` is set on frames fitted to given poses (`ModelFrame.fit`).
    """

    anchor_c2w: torch.Tensor  # [B, 4, 4], world frame
    scale: float | torch.Tensor
    residuals: PoseResiduals | None = None

    @classmethod
    def fit(cls, predicted_c2w: torch.Tensor, given_c2w: torch.Tensor) -> ModelFrame:
        """The frame that maps a model's predicted cameras `predicted_c2w` onto
        the caller's `given_c2w` (both [B, V, 4, 4], the same views).

        Closed form over all V >= 2 views, in fp64: the rotation is the chordal
        mean of R_given R_predicted^T (SVD, projected to det +1); with it fixed,
        scale and translation are the least-squares fit of the camera centres.
        """
        pred, given = predicted_c2w.double(), given_c2w.to(predicted_c2w).double()
        if pred.shape != given.shape:
            raise ValueError(
                f"predicted cameras {tuple(pred.shape)} and given c2w "
                f"{tuple(given.shape)} disagree"
            )
        if pred.shape[1] < 2:
            raise ValueError(
                f"fitting a frame to given poses needs at least 2 views, got "
                f"{pred.shape[1]}"
            )
        _check_rotation(given[..., :3, :3], "given c2w")
        _check_rotation(pred[..., :3, :3], "predicted c2w")
        for what, c2w in (("given", given), ("predicted", pred)):
            if not torch.isfinite(c2w[..., :3, 3]).all():
                raise ValueError(f"{what} c2w: the camera centres are not finite")
        r_pred, r_given = pred[..., :3, :3], given[..., :3, :3]
        c_pred, c_given = pred[..., :3, 3], given[..., :3, 3]
        u, _, vh = torch.linalg.svd((r_given @ r_pred.transpose(-1, -2)).sum(dim=1))
        flip = torch.ones_like(u[..., 0])
        flip[..., -1] = torch.sign(torch.linalg.det(u @ vh))
        rotation = u @ torch.diag_embed(flip) @ vh  # [B, 3, 3]

        d_pred = c_pred - c_pred.mean(dim=1, keepdim=True)
        d_given = c_given - c_given.mean(dim=1, keepdim=True)
        spread = (d_pred**2).sum(dim=(1, 2))
        extent = (d_given**2).sum(dim=-1).mean(dim=1).sqrt()  # RMS, given
        if (spread <= 1e-12 * (c_pred**2).sum(dim=(1, 2)).clamp_min(1e-300)).any():
            raise ValueError("the predicted camera centres coincide: no scale to fit")
        rotated = torch.einsum("bij,bvj->bvi", rotation, d_pred)
        scale = (d_given * rotated).sum(dim=(1, 2)) / spread  # model -> world
        if (scale <= 0).any():
            raise ValueError(
                "the given poses do not fit the predicted cameras (fitted scale "
                f"{scale.tolist()} <= 0)"
            )
        translation = c_given.mean(dim=1) - scale[:, None] * torch.einsum(
            "bij,bj->bi", rotation, c_pred.mean(dim=1)
        )

        delta = r_given @ (rotation[:, None] @ r_pred).transpose(-1, -2)
        cos = (delta.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
        skew = delta - delta.transpose(-1, -2)  # atan2: accurate at small angles
        sin = torch.stack([skew[..., 2, 1], skew[..., 0, 2], skew[..., 1, 0]], -1)
        sin = sin.norm(dim=-1) / 2
        mapped_c = scale[:, None, None] * torch.einsum("bij,bvj->bvi", rotation, c_pred)
        error = (mapped_c + translation[:, None] - c_given).norm(dim=-1)
        residuals = PoseResiduals(
            rotation_deg=torch.rad2deg(torch.atan2(sin, cos)).float(),
            center_error=error.float(),
            relative_center_error=(error / extent[:, None].clamp_min(1e-300)).float(),
        )
        anchor = torch.eye(4, dtype=torch.float64, device=pred.device)
        anchor = anchor.repeat(pred.shape[0], 1, 1)
        anchor[:, :3, :3] = rotation
        anchor[:, :3, 3] = translation
        return cls(
            anchor_c2w=anchor.float(),
            scale=(1 / scale).float(),
            residuals=residuals,
        )

    def c2w_to_model(self, c2w: torch.Tensor) -> torch.Tensor:
        """World c2w [B, V, 4, 4] -> model frame.

        The dataset's arithmetic, op for op (`camera_normalization` followed by the
        in-place translation scale), so the result is bit-identical to training.
        """
        if c2w.shape[0] != self.anchor_c2w.shape[0]:
            raise ValueError(
                f"{c2w.shape[0]} camera samples for a frame of "
                f"{self.anchor_c2w.shape[0]}"
            )
        out = []
        for anchor, poses in zip(self.anchor_c2w, c2w):
            canonical = torch.eye(4, dtype=torch.float32, device=anchor.device)[None]
            norm = torch.bmm(canonical, torch.inverse(anchor[None]))
            out.append(torch.bmm(norm.repeat(poses.shape[0], 1, 1), poses))
        model = torch.stack(out)
        if isinstance(self.scale, torch.Tensor):
            model[..., :3, 3] *= self.scale.to(model)[:, None, None]
        elif self.scale != 1.0:
            model[..., :3, 3] *= self.scale
        return model

    def cameras_to_model(self, cameras: Cameras) -> Cameras:
        """World-frame cameras -> model frame (fp32, on the frame's device)."""
        c2w = cameras.c2w.float().to(self.anchor_c2w.device)
        return replace(cameras, c2w=self.c2w_to_model(c2w))

    def model_to_world(self) -> Similarity:
        """model -> world: x_w = anchor_c2w @ (x_m / scale)."""
        anchor = self.anchor_c2w
        if isinstance(self.scale, torch.Tensor):
            scale = 1.0 / self.scale.to(anchor)
        else:
            scale = torch.full_like(anchor[:, 0, 0], 1.0 / self.scale)
        return Similarity(
            rotation=anchor[:, :3, :3], scale=scale, translation=anchor[:, :3, 3]
        )


@dataclass
class Similarity:
    """x -> scale * rotation @ x + translation, per sample.

    The one transform every model's Gaussians go through between frames (the
    public entry is `GSOutput.to_world`). Only rotations, uniform scales and
    translations keep a Gaussian a Gaussian with the same opacity and its colour
    an SH field, so anything else is rejected.
    """

    rotation: torch.Tensor  # [B, 3, 3], proper (det +1)
    scale: torch.Tensor  # [B], > 0
    translation: torch.Tensor  # [B, 3]

    def __post_init__(self) -> None:
        _check_rotation(self.rotation, "not a similarity")
        if not (torch.isfinite(self.scale) & (self.scale > 0)).all():
            raise ValueError(f"scale must be finite and > 0, got {self.scale}")
        if not torch.isfinite(self.translation).all():
            raise ValueError(f"translation must be finite, got {self.translation}")

    def apply_cameras(self, cameras: Cameras) -> Cameras:
        """Camera poses moved with the scene: rotations by `rotation`, centres
        as points. Intrinsics are unchanged."""
        c2w = cameras.c2w
        rotation = self.rotation.to(c2w)[:, None]
        out = c2w.clone()
        out[..., :3, :3] = rotation @ c2w[..., :3, :3]
        centers = (c2w[..., :3, 3] * self.scale.to(c2w)[:, None, None])[..., None]
        out[..., :3, 3] = (rotation @ centers)[..., 0]
        out[..., :3, 3] += self.translation.to(c2w)[:, None]
        return replace(cameras, c2w=out)

    def apply(self, gaussians: Gaussians) -> Gaussians:
        """Each attribute as the transform acts on it: means fully, scales by
        `scale`, quats by `rotation`, SH of degree >= 1 rotated; opacities and RGB /
        degree-0 colours unchanged. Computed in the dtype of `gaussians`."""
        g = gaussians
        rotation = self.rotation.to(g.means)
        scale = self.scale.to(g.means)
        means = (g.means * scale[:, None, None]) @ rotation.transpose(-1, -2)
        means = means + self.translation.to(g.means)[:, None]
        quats = quat_multiply(matrix_to_quat(rotation)[:, None], g.quats)
        colors = g.colors
        if g.sh_degree is not None and g.sh_degree >= 1:
            colors = rotate_sh(colors, rotation, g.sh_degree)
        return replace(
            g,
            means=means,
            scales=g.scales * scale[:, None, None],
            quats=quats,
            colors=colors,
        )


def anchored_frame(views_c2w: torch.Tensor, scale: float) -> ModelFrame:
    """The frame anchored to the first context view of each sample."""
    return ModelFrame(anchor_c2w=views_c2w[:, 0].float(), scale=float(scale))


def matrix_to_quat(rotation: torch.Tensor) -> torch.Tensor:
    """[.., 3, 3] rotation -> [.., 4] unit quaternion (wxyz, w >= 0)."""
    m = rotation
    m00, m11, m22 = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    m01, m02, m10 = m[..., 0, 1], m[..., 0, 2], m[..., 1, 0]
    m12, m20, m21 = m[..., 1, 2], m[..., 2, 0], m[..., 2, 1]
    # Shepperd: take the largest of |w|, |x|, |y|, |z| from the diagonal and the
    # other three from off-diagonal sums divided by it -- accurate for every
    # rotation, unlike sqrt-ing each component (lossy near 0) or the trace-only
    # formula (breaks near 180 degrees).
    squares = torch.stack(
        (
            1 + m00 + m11 + m22,
            1 + m00 - m11 - m22,
            1 - m00 + m11 - m22,
            1 - m00 - m11 + m22,
        ),
        dim=-1,
    )
    root = torch.sqrt(squares.clamp_min(1e-12))  # 2 * |component|
    candidates = torch.stack(
        (
            torch.stack((root[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), -1),
            torch.stack((m21 - m12, root[..., 1] ** 2, m10 + m01, m02 + m20), -1),
            torch.stack((m02 - m20, m10 + m01, root[..., 2] ** 2, m12 + m21), -1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, root[..., 3] ** 2), -1),
        ),
        dim=-2,
    ) / (2 * root[..., None])
    best = squares.argmax(dim=-1)
    q = torch.gather(
        candidates, -2, best[..., None, None].expand(*best.shape, 1, 4)
    ).squeeze(-2)
    q = q / q.norm(dim=-1, keepdim=True)
    return torch.where(q[..., :1] < 0, -q, q)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """[.., 4] quaternion (wxyz, any norm) -> [.., 3, 3] rotation."""
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


def quat_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a * b (wxyz): rotate by b, then by a."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def rotate_sh(
    colors: torch.Tensor, rotation: torch.Tensor, degree: int
) -> torch.Tensor:
    """Rotate SH coefficients [B, N, K, 3] by per-sample rotations [B, 3, 3].

    The rotated field along d is the original along R^T d, in the real SH basis of
    3DGS / gsplat, any degree. Degree 0 is rotation invariant. Degree 1 is
    C1 * (-y c1 + z c2 - x c3), a linear form a . d with a = (-c3, -c1, c2), so it
    turns into R a. Degrees >= 2 use `sh_rotation_blocks`, built in fp64.
    """
    if colors.shape[-2] != (degree + 1) ** 2:
        raise ValueError(
            f"degree {degree} needs {(degree + 1) ** 2} SH coefficients, got "
            f"{tuple(colors.shape)}"
        )
    rotated = colors.clone()
    if degree < 1:
        return rotated
    c1, c2, c3 = colors[:, :, 1], colors[:, :, 2], colors[:, :, 3]
    a = torch.stack((-c3, -c1, c2), dim=2)  # [B, N, xyz, rgb]
    a = torch.einsum("bij,bnjc->bnic", rotation, a)
    rotated[:, :, 1] = -a[:, :, 1]
    rotated[:, :, 2] = a[:, :, 2]
    rotated[:, :, 3] = -a[:, :, 0]
    if degree < 2:
        return rotated
    blocks = sh_rotation_blocks(rotation.double(), degree)
    for degree_l in range(2, degree + 1):
        band = slice(degree_l**2, (degree_l + 1) ** 2)
        block = blocks[degree_l].to(colors.dtype)
        rotated[:, :, band] = torch.einsum("bij,bnjc->bnic", block, colors[:, :, band])
    return rotated


def sh_rotation_blocks(rotation: torch.Tensor, degree: int) -> list[torch.Tensor]:
    """Real SH rotation matrices D_l [.., 2l+1, 2l+1] for l = 0..degree, in the
    3DGS / gsplat basis: coefficients c_l of degree l become D_l @ c_l.

    Ivanic & Ruedenberg's recursion (J. Phys. Chem. 100, 6342 (1996); errata 102,
    9099 (1998)), exact for any degree: D_l from D_{l-1} and D_1, which is R in the
    (y, z, x) order of the real basis without the Condon-Shortley phase. gsplat's
    basis carries that phase, (-1)^m on order m, so each D_l is conjugated by it.
    """
    order = [1, 2, 0]  # m = -1, 0, 1 <-> y, z, x
    r1 = rotation[..., order, :][..., :, order]
    blocks = [torch.ones_like(rotation[..., :1, :1]), r1]
    for degree_l in range(2, degree + 1):
        blocks.append(_ivanic_ruedenberg_step(r1, blocks[-1], degree_l))
    out = []
    for degree_l, block in enumerate(blocks[: degree + 1]):
        m = torch.arange(-degree_l, degree_l + 1, device=rotation.device)
        phase = (1 - 2 * (m % 2)).to(rotation.dtype)
        out.append(block * phase[:, None] * phase[None, :])
    return out


def _ivanic_ruedenberg_step(
    r1: torch.Tensor, prev: torch.Tensor, degree: int
) -> torch.Tensor:
    """D_l [.., 2l+1, 2l+1] from D_1 and D_{l-1}, indexed by order m in -l..l."""
    lv = degree

    def d1(i: int, j: int) -> torch.Tensor:
        return r1[..., i + 1, j + 1]

    def d_prev(a: int, b: int) -> torch.Tensor:
        return prev[..., a + lv - 1, b + lv - 1]

    def p(i: int, a: int, b: int) -> torch.Tensor:
        if b == lv:
            return d1(i, 1) * d_prev(a, lv - 1) - d1(i, -1) * d_prev(a, -lv + 1)
        if b == -lv:
            return d1(i, 1) * d_prev(a, -lv + 1) + d1(i, -1) * d_prev(a, lv - 1)
        return d1(i, 0) * d_prev(a, b)

    def u(m: int, n: int) -> torch.Tensor:
        return p(0, m, n)

    def v(m: int, n: int) -> torch.Tensor:
        if m == 0:
            return p(1, 1, n) + p(-1, -1, n)
        if m > 0:
            first = p(1, m - 1, n) * math.sqrt(1 + (m == 1))
            return first if m == 1 else first - p(-1, -m + 1, n)
        first = p(-1, -m - 1, n) * math.sqrt(1 + (m == -1))
        return first if m == -1 else first + p(1, m + 1, n)

    def w(m: int, n: int) -> torch.Tensor:
        if m > 0:
            return p(1, m + 1, n) + p(-1, -m - 1, n)
        return p(1, m - 1, n) - p(-1, -m + 1, n)

    rows = []
    for m in range(-lv, lv + 1):
        row = []
        for n in range(-lv, lv + 1):
            am = abs(m)
            denom = 2 * lv * (2 * lv - 1) if abs(n) == lv else (lv + n) * (lv - n)
            cu = math.sqrt((lv + m) * (lv - m) / denom)
            cv = 0.5 * math.sqrt((1 + (m == 0)) * (lv + am - 1) * (lv + am) / denom)
            cv = -cv if m == 0 else cv
            cw = 0.0 if m == 0 else -0.5 * math.sqrt((lv - am - 1) * (lv - am) / denom)
            entry = cv * v(m, n)
            if cu:
                entry = entry + cu * u(m, n)
            if cw:
                entry = entry + cw * w(m, n)
            row.append(entry)
        rows.append(torch.stack(row, dim=-1))
    return torch.stack(rows, dim=-2)
