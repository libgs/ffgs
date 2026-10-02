"""Models that predict their own cameras: optional poses, fitted frames."""

import pytest
import torch
from _dummy import (
    PoseFreeDummyModel,
    PoseFreeDummyProcessor,
)
from test_geometry import _random_c2w
from test_pipeline import _fake_render

from ffgs import Cameras, GSPipeline, ModelFrame, Views
from ffgs.geometry import Similarity, quat_to_matrix


def _pipe() -> GSPipeline:
    torch.manual_seed(0)
    return GSPipeline(PoseFreeDummyModel(), PoseFreeDummyProcessor())


def _images(b: int = 1, v: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, 256, (b, v, 3, 24, 40), dtype=torch.uint8, generator=g)


def _known_similarity(b: int) -> Similarity:
    g = torch.Generator().manual_seed(2)
    return Similarity(
        rotation=quat_to_matrix(torch.randn(b, 4, generator=g, dtype=torch.float64)),
        scale=torch.tensor([2.5, 0.4][:b], dtype=torch.float64),
        translation=torch.randn(b, 3, generator=g, dtype=torch.float64) * 3,
    )


def _posed(images: torch.Tensor, c2w: torch.Tensor) -> Views:
    return Views(images, c2w=c2w)


def test_without_poses_there_is_no_world() -> None:
    pipe = _pipe()
    out = pipe(Views(_images()))
    assert out.frame is None and out.space == "model"
    assert out.cameras.c2w.shape == (1, 4, 4, 4)
    assert out.gaussians.sh_degree == 2
    with pytest.raises(ValueError, match="no world frame"):
        out.to_world()


def test_given_poses_recover_a_known_similarity() -> None:
    pipe, images = _pipe(), _images(b=2)
    predicted = pipe(Views(images)).cameras
    known = _known_similarity(2)
    given = known.apply_cameras(replace_c2w(predicted, predicted.c2w.double())).c2w

    out = pipe(_posed(images, given.float()))
    assert torch.equal(out.cameras.c2w, predicted.c2w)  # poses do not enter the model
    fitted = out.frame.model_to_world()
    torch.testing.assert_close(
        fitted.rotation.double(), known.rotation, atol=1e-6, rtol=0
    )
    torch.testing.assert_close(fitted.scale.double(), known.scale, atol=0, rtol=1e-5)
    torch.testing.assert_close(
        fitted.translation.double(), known.translation, atol=1e-5, rtol=0
    )
    residuals = out.frame.residuals
    assert residuals.rotation_deg.shape == residuals.center_error.shape == (2, 4)
    assert residuals.rotation_deg.max() < 1e-3
    assert residuals.relative_center_error.max() < 1e-5

    # In fp64 the fit is exact.
    frame = ModelFrame.fit(predicted.c2w.double(), given)
    assert frame.residuals.rotation_deg.max() < 1e-6
    assert frame.residuals.relative_center_error.max() < 1e-12

    # to_world takes the predicted cameras onto the given ones, and the frame
    # takes the given ones back.
    world = out.to_world()
    torch.testing.assert_close(world.cameras.c2w, given.float(), atol=1e-4, rtol=0)
    torch.testing.assert_close(
        out.frame.c2w_to_model(given.float()), predicted.c2w, atol=1e-4, rtol=0
    )
    expected = known.apply(out.gaussians.to(dtype=torch.float64))
    torch.testing.assert_close(
        world.gaussians.means.double(), expected.means, atol=1e-4, rtol=1e-5
    )
    torch.testing.assert_close(
        world.gaussians.colors.double(), expected.colors, atol=1e-5, rtol=1e-5
    )


def replace_c2w(cameras: Cameras, c2w: torch.Tensor) -> Cameras:
    return Cameras(c2w, cameras.intrinsics, cameras.image_shape, True)


def test_noisy_poses_leave_residuals() -> None:
    pipe, images = _pipe(), _images(v=6)
    predicted = pipe(Views(images)).cameras
    given = (
        _known_similarity(1)
        .apply_cameras(replace_c2w(predicted, predicted.c2w.double()))
        .c2w
    )
    g = torch.Generator().manual_seed(3)
    tilt = torch.randn(1, 6, 3, generator=g, dtype=torch.float64) * 0.03  # ~2 deg
    tilt = torch.cat([torch.ones_like(tilt[..., :1]), tilt / 2], dim=-1)
    noisy = given.clone()
    noisy[..., :3, :3] = quat_to_matrix(tilt) @ given[..., :3, :3]
    noisy[..., :3, 3] += torch.randn(1, 6, 3, generator=g, dtype=torch.float64) * 0.05

    residuals = pipe(_posed(images, noisy.float())).frame.residuals
    assert 0.3 < residuals.rotation_deg.mean() < 5
    assert (residuals.center_error > 0).all()
    assert 0.01 < residuals.center_error.mean() < 0.2
    assert residuals.relative_center_error.mean() > 1e-3


def test_fit_needs_two_views_and_distinct_centres() -> None:
    g = torch.Generator().manual_seed(4)
    c2w = _random_c2w(g, 3)[None]
    with pytest.raises(ValueError, match="at least 2 views"):
        ModelFrame.fit(c2w[:, :1], c2w[:, :1])
    with pytest.raises(ValueError, match="disagree"):
        ModelFrame.fit(c2w, c2w[:, :2])
    same = c2w.clone()
    same[..., :3, 3] = 1.0
    with pytest.raises(ValueError, match="coincide"):
        ModelFrame.fit(same, c2w)
    flipped = c2w.clone()
    flipped[..., :3, 3] *= -1  # centres mirrored: no positive scale fits
    with pytest.raises(ValueError, match="<= 0"):
        ModelFrame.fit(c2w, flipped)


@pytest.mark.parametrize(
    "factors, match",
    [
        ((1.0, 1.0, -1.0), "det < 0"),  # reflection: its residual angle reads 0
        ((1.0, 1.0, 2.0), "not orthonormal"),
    ],
)
def test_fit_rejects_cameras_that_are_not_rigid(factors, match) -> None:
    g = torch.Generator().manual_seed(4)
    c2w = _random_c2w(g, 3)[None]
    bad = c2w.clone()
    bad[..., :3, :3] = bad[..., :3, :3] @ torch.diag(torch.tensor(factors))
    with pytest.raises(ValueError, match=f"given c2w: .*{match}"):
        ModelFrame.fit(c2w, bad)
    with pytest.raises(ValueError, match=f"predicted c2w: .*{match}"):
        ModelFrame.fit(bad, c2w)


def test_fit_rejects_cameras_that_are_not_finite() -> None:
    g = torch.Generator().manual_seed(4)
    c2w = _random_c2w(g, 3)[None]
    bad = c2w.clone()
    bad[0, 1, 0, 3] = float("nan")
    with pytest.raises(ValueError, match="given c2w: the camera centres"):
        ModelFrame.fit(c2w, bad)
    with pytest.raises(ValueError, match="predicted c2w: the camera centres"):
        ModelFrame.fit(bad, c2w)


def test_one_posed_view_cannot_fix_a_frame() -> None:
    pipe = _pipe()
    with pytest.raises(ValueError, match="at least 2 views"):
        pipe(_posed(_images(v=1), torch.eye(4)[None, None]))


def test_render_is_the_same_with_or_without_a_frame(monkeypatch) -> None:
    monkeypatch.setattr("ffgs.pipeline.render_mod.render", _fake_render)
    pipe, images = _pipe(), _images(b=2)
    bare = pipe(Views(images))
    given = (
        _known_similarity(2)
        .apply_cameras(replace_c2w(bare.cameras, bare.cameras.c2w.double()))
        .c2w.float()
    )
    posed = pipe(_posed(images, given))
    world = posed.to_world()

    # No frame: cameras and near / far are in the model's frame.
    at_bare = pipe.render(bare, bare.cameras)
    # Fitted frame: cameras in the world, Gaussians in either space.
    world_cameras = Cameras(
        given, bare.cameras.intrinsics, bare.cameras.image_shape, True
    )
    at_model = pipe.render(posed, world_cameras)
    at_world = pipe.render(world, world_cameras)
    at_world_predicted = pipe.render(world, world.cameras)
    for key, value in at_model.items():
        torch.testing.assert_close(value, at_world[key], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(value, at_bare[key], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(value, at_world_predicted[key], atol=1e-4, rtol=1e-4)


def test_pose_free_round_trip(tmp_path) -> None:
    pipe = _pipe()
    pipe.save_pretrained(tmp_path)
    loaded = GSPipeline.from_pretrained(tmp_path)
    views = Views(_images())
    assert torch.equal(loaded(views).cameras.c2w, pipe(views).cameras.c2w)
