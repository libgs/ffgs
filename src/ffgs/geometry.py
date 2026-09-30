"""The model's working frame, and moving cameras / Gaussians in and out of it.

Feed-forward models are trained on normalised cameras: here, every pose relative to
the first context view, then translations scaled by a constant `scene_scale`. A
`ModelFrame` records that normalisation per sample, so inputs go in with exactly the
training arithmetic and predicted Gaussians come back in the caller's world frame.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from .types import Cameras, Gaussians

_SH_C1_DEGREE = 1


@dataclass
class ModelFrame:
    """world -> model: x_m = scale * inverse(anchor_c2w) @ x_w, per sample."""

    anchor_c2w: torch.Tensor  # [B, 4, 4], world frame
    scale: float

    def c2w_to_model(self, c2w: torch.Tensor) -> torch.Tensor:
        """World c2w [B, V, 4, 4] -> model frame.

        The dataset's arithmetic, op for op (`camera_normalization` followed by the
        in-place translation scale), so the result is bit-identical to training.
        """
        out = []
        for anchor, poses in zip(self.anchor_c2w, c2w):
            canonical = torch.eye(4, dtype=torch.float32, device=anchor.device)[None]
            norm = torch.bmm(canonical, torch.inverse(anchor[None]))
            out.append(torch.bmm(norm.repeat(poses.shape[0], 1, 1), poses))
        model = torch.stack(out)
        if self.scale != 1.0:
            model[..., :3, 3] *= self.scale
        return model

    def cameras_to_model(self, cameras: Cameras) -> Cameras:
        return replace(cameras, c2w=self.c2w_to_model(cameras.c2w))

    def gaussians_to_world(self, gaussians: Gaussians) -> Gaussians:
        """Model-frame Gaussians -> world frame (fp32)."""
        g = gaussians.to(dtype=torch.float32)
        anchor = self.anchor_c2w.to(g.means)
        rotation, translation = anchor[:, :3, :3], anchor[:, :3, 3]
        means = (g.means / self.scale) @ rotation.transpose(-1, -2)
        means = means + translation[:, None]
        quats = quat_multiply(matrix_to_quat(rotation)[:, None], g.quats)
        colors = g.colors
        if g.sh_degree is not None and g.sh_degree >= _SH_C1_DEGREE:
            colors = rotate_sh(colors, rotation, g.sh_degree)
        return replace(
            g,
            means=means,
            scales=g.scales / self.scale,
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

    Degree 1 in the 3DGS / gsplat basis is C1 * (-y c1 + z c2 - x c3), i.e. a linear
    form a . d with a = (-c3, -c1, c2). Rotating the field by R turns a into R a.
    Degree 0 is rotation invariant.
    """
    if degree > 1:
        raise NotImplementedError(f"SH rotation for degree {degree} is not needed yet")
    c1, c2, c3 = colors[:, :, 1], colors[:, :, 2], colors[:, :, 3]
    a = torch.stack((-c3, -c1, c2), dim=2)  # [B, N, xyz, rgb]
    a = torch.einsum("bij,bnjc->bnic", rotation, a)
    rotated = colors.clone()
    rotated[:, :, 1] = -a[:, :, 1]
    rotated[:, :, 2] = a[:, :, 2]
    rotated[:, :, 3] = -a[:, :, 0]
    return rotated
