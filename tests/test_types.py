import pytest
import torch
from plyfile import PlyData
from test_geometry import random_gaussians

from ffgs import Cameras, Gaussians, Views
from ffgs.types import SH_C0, read_ply_meta


def test_save_ply_writes_the_3dgs_layout(tmp_path) -> None:
    g = torch.Generator().manual_seed(4)
    gs = random_gaussians(g, 1, 20, sh_degree=1)
    path = gs.save_ply(tmp_path / "scene.ply")
    vertex = PlyData.read(str(path))["vertex"]
    names = [p.name for p in vertex.properties]
    assert names[:9] == ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    assert [n for n in names if n.startswith("f_rest_")] == [
        f"f_rest_{i}" for i in range(9)
    ]
    assert names[-8:] == [
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    assert len(vertex) == 20
    opacity = torch.sigmoid(torch.tensor(vertex["opacity"]))
    torch.testing.assert_close(opacity, gs.opacities[0], atol=1e-5, rtol=0)
    # f_rest is channel-major: f_rest_0..2 are the red degree-1 coefficients.
    torch.testing.assert_close(
        torch.tensor(vertex["f_rest_1"]), gs.colors[0, :, 2, 0], atol=0, rtol=0
    )
    torch.testing.assert_close(
        torch.exp(torch.tensor(vertex["scale_2"])), gs.scales[0, :, 2]
    )


def test_save_ply_prunes_zero_opacity(tmp_path) -> None:
    g = torch.Generator().manual_seed(5)
    gs = random_gaussians(g, 1, 10, sh_degree=1)
    gs.opacities[0, 3:] = 0
    vertex = PlyData.read(str(gs.save_ply(tmp_path / "pruned.ply")))["vertex"]
    assert len(vertex) == 3
    torch.testing.assert_close(torch.tensor(vertex["x"]), gs.means[0, :3, 0])
    path = gs.save_ply(tmp_path / "all.ply", prune=False)
    vertex = PlyData.read(str(path))["vertex"]
    assert len(vertex) == 10
    opacity = torch.sigmoid(torch.tensor(vertex["opacity"]))
    torch.testing.assert_close(opacity[3:], torch.full((7,), 1e-6), atol=1e-9, rtol=0)


def test_rgb_gaussians_save_as_degree_zero(tmp_path) -> None:
    g = torch.Generator().manual_seed(6)
    gs = random_gaussians(g, 1, 8, sh_degree=None)
    vertex = PlyData.read(str(gs.save_ply(tmp_path / "rgb.ply")))["vertex"]
    assert not [p.name for p in vertex.properties if p.name.startswith("f_rest_")]
    rgb = torch.tensor(vertex["f_dc_0"]) * SH_C0 + 0.5
    torch.testing.assert_close(rgb, gs.colors[0, :, 0])


def test_gaussians_validate_color_shape() -> None:
    with pytest.raises(ValueError, match="SH degree 1"):
        Gaussians(
            means=torch.zeros(1, 2, 3),
            scales=torch.ones(1, 2, 3),
            quats=torch.zeros(1, 2, 4),
            opacities=torch.ones(1, 2),
            colors=torch.zeros(1, 2, 3),
            sh_degree=1,
        )


def test_render_cameras_accept_unbatched_input() -> None:
    cams = Cameras(torch.eye(4).repeat(2, 1, 1), torch.eye(3).repeat(2, 1, 1), (8, 8))
    assert cams.c2w.shape == (1, 2, 4, 4)
    assert cams.image_shape == (8, 8)


@pytest.mark.parametrize("view", [0, 1])  # the anchor view and a later one
def test_cameras_and_views_reject_nonfinite_poses(view) -> None:
    images, k = torch.zeros(2, 3, 8, 8), torch.eye(3).repeat(2, 1, 1)
    c2w = torch.eye(4).repeat(2, 1, 1)
    c2w[view, 0, 3] = float("nan")
    with pytest.raises(ValueError, match="c2w has nonfinite"):
        Cameras(c2w, k, (8, 8))
    with pytest.raises(ValueError, match="c2w has nonfinite"):
        Views(images, k, c2w)
    k[view, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="intrinsics has nonfinite"):
        Views(images, k, torch.eye(4).repeat(2, 1, 1))


def test_cameras_and_views_reject_skew() -> None:
    k = torch.eye(3).repeat(2, 1, 1) * 100
    k[:, 2, 2] = 1
    k[1, 0, 1] = 5.0
    with pytest.raises(ValueError, match="skew"):
        Cameras(torch.eye(4).repeat(2, 1, 1), k, (8, 8))
    with pytest.raises(ValueError, match="skew"):
        Views(torch.zeros(2, 3, 8, 8), k, torch.eye(4).repeat(2, 1, 1))
    k[1, 0, 1] = 1e-7  # numerical noise passes
    Cameras(torch.eye(4).repeat(2, 1, 1), k, (8, 8))


def test_views_convert_uint8_and_check_shapes() -> None:
    images = torch.full((2, 3, 4, 6), 255, dtype=torch.uint8)
    views = Views(images, torch.eye(3).repeat(2, 1, 1), torch.eye(4).repeat(2, 1, 1))
    assert views.images.dtype == torch.float32
    assert views.images.shape == (1, 2, 3, 4, 6) and views.images.max() == 1.0
    assert views.image_shape == (4, 6)
    with pytest.raises(ValueError, match="disagree"):
        Views(images, torch.eye(3).repeat(3, 1, 1), torch.eye(4).repeat(2, 1, 1))
    with pytest.raises(ValueError, match="RGB"):
        Views(images[:, :1], torch.eye(3).repeat(2, 1, 1), torch.eye(4).repeat(2, 1, 1))


def test_views_poses_and_intrinsics_are_optional() -> None:
    images = torch.zeros(2, 3, 4, 6)
    bare = Views(images)
    assert bare.c2w is None and bare.intrinsics is None and not bare.has_poses
    with pytest.raises(ValueError, match="Views.cameras needs c2w and intrinsics"):
        bare.cameras
    with pytest.raises(ValueError, match="needs intrinsics; these views have no"):
        bare.normalized_k
    posed = Views(images, c2w=torch.eye(4).repeat(2, 1, 1))
    assert posed.has_poses and posed.intrinsics is None
    with pytest.raises(ValueError, match="no intrinsics"):
        posed.require("c2w", "intrinsics", by="X")
    with pytest.raises(ValueError, match="disagree"):
        Views(images, c2w=torch.eye(4).repeat(3, 1, 1))


def test_save_ply_writes_meta_as_header_comments(tmp_path) -> None:
    g = torch.Generator().manual_seed(7)
    gs = random_gaussians(g, 1, 12, sh_degree=1)
    plain = gs.save_ply(tmp_path / "plain.ply").read_bytes()
    meta = {"version": "1.2.3", "model": "family/variant", "background": "1 1 1"}
    data = gs.save_ply(tmp_path / "meta.ply", meta=meta).read_bytes()
    header, body = data.split(b"end_header\n", 1)
    assert header.decode("ascii").split("\n")[:5] == [
        "ply",
        "format binary_little_endian 1.0",
        "comment ffgs-version: 1.2.3",
        "comment ffgs-model: family/variant",
        "comment ffgs-background: 1 1 1",
    ]
    assert body == plain.split(b"end_header\n", 1)[1]
    assert read_ply_meta(tmp_path / "meta.ply") == meta
    assert PlyData.read(str(tmp_path / "meta.ply"))["vertex"].count == 12


def test_read_ply_meta_skips_other_comments(tmp_path) -> None:
    path = tmp_path / "other.ply"
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\ncomment Generated by a tool 1.0\n"
        b"comment ffgs-space: world\ncomment ffgs-broken\nobj_info ffgs-model: x\n"
        b"element vertex 0\nproperty float x\nend_header\n"
    )
    assert read_ply_meta(path) == {"space": "world"}
    g = torch.Generator().manual_seed(8)
    assert (
        read_ply_meta(
            random_gaussians(g, 1, 4, sh_degree=None).save_ply(tmp_path / "p.ply")
        )
        == {}
    )
    (tmp_path / "x.txt").write_text("not a ply\n")
    with pytest.raises(ValueError, match="not a ply"):
        read_ply_meta(tmp_path / "x.txt")
    (tmp_path / "cut.ply").write_bytes(b"ply\nformat ascii 1.0\n")
    with pytest.raises(ValueError, match="end_header"):
        read_ply_meta(tmp_path / "cut.ply")


@pytest.mark.parametrize(
    "meta, match",
    [
        ({"Model": "x"}, "key"),
        ({"model_name": "x"}, "key"),
        ({"model": "a\nb"}, "one line"),
        ({"model": " x"}, "one line"),
        ({"model": "模型"}, "one line"),
        ({"model": ""}, "empty"),
    ],
)
def test_save_ply_rejects_bad_meta(tmp_path, meta, match) -> None:
    g = torch.Generator().manual_seed(9)
    with pytest.raises(ValueError, match=match):
        random_gaussians(g, 1, 4, sh_degree=None).save_ply(
            tmp_path / "bad.ply", meta=meta
        )
    assert not (tmp_path / "bad.ply").exists()
