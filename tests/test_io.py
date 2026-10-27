"""`ffgs.io`: camera files (transforms.json, COLMAP, cameras.json) and pairing."""

import ast
import json
import math
import struct
import warnings
from pathlib import Path

import pytest
import torch
from PIL import Image
from test_examples import _scene

from ffgs import io
from ffgs.geometry import quat_to_matrix
from ffgs.io import Camera, pair_cameras, read_cameras, save_cameras

# --- cameras ------------------------------------------------------------------------

GL_TO_CV = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64))
QVECS = [(1.0, 0.0, 0.0, 0.0), (0.9, 0.1, -0.3, 0.2), (0.5, -0.5, 0.5, 0.5)]
TVECS = [(0.0, 0.0, 0.0), (0.5, -1.0, 2.0), (-0.2, 0.3, 4.0)]


def _c2w(qvec, tvec) -> torch.Tensor:
    """OpenCV c2w of a COLMAP world-to-camera pose."""
    rotation = quat_to_matrix(torch.tensor(qvec, dtype=torch.float64))
    c2w = torch.eye(4, dtype=torch.float64)
    c2w[:3, :3] = rotation.T
    c2w[:3, 3] = -rotation.T @ torch.tensor(tvec, dtype=torch.float64)
    return c2w


def _assert_same(a: list[Camera], b: list[Camera]) -> None:
    assert [c.name for c in a] == [c.name for c in b]
    for x, y in zip(a, b):
        assert x.size == y.size
        torch.testing.assert_close(x.k, y.k)
        torch.testing.assert_close(x.c2w, y.c2w)


def _write_colmap(root, cameras, images, ext) -> None:
    """cameras: [(id, model, w, h, params)]; images: [(id, qvec, tvec, camera, name)]."""
    root.mkdir(parents=True, exist_ok=True)
    if ext == "txt":
        lines = ["# Camera list"]
        lines += [
            f"{i} {m} {w} {h} " + " ".join(map(str, p)) for i, m, w, h, p in cameras
        ]
        (root / "cameras.txt").write_text("\n".join(lines) + "\n")
        lines = ["# Image list"]
        for i, q, t, cam, name in images:
            lines += [" ".join(map(str, (i, *q, *t, cam, name))), ""]
        (root / "images.txt").write_text("\n".join(lines) + "\n")
        return
    ids = {name: i for i, (name, _) in io.COLMAP_MODELS.items()}
    data = struct.pack("<Q", len(cameras))
    for i, m, w, h, p in cameras:
        data += struct.pack("<iiQQ", i, ids[m], w, h) + struct.pack(f"<{len(p)}d", *p)
    (root / "cameras.bin").write_bytes(data)
    data = struct.pack("<Q", len(images))
    for i, q, t, cam, name in images:
        data += struct.pack("<i7di", i, *q, *t, cam) + name.encode() + b"\0"
        data += struct.pack("<Q", 1) + struct.pack("<ddq", 1.0, 2.0, -1)
    (root / "images.bin").write_bytes(data)


def _transforms(path, meta) -> Path:
    path.write_text(json.dumps(meta))
    return path


def test_io_is_independent_of_the_core() -> None:
    # Only reads and writes files: no import of ffgs's own modules.
    tree = ast.parse(Path(io.__file__).read_text())
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert all(n.level == 0 and not n.module.startswith("ffgs") for n in imports)


def test_camera_checks() -> None:
    k, c2w = torch.eye(3), torch.eye(4)
    camera = Camera("a.png", k.tolist(), c2w.numpy(), (40.0, 24))
    assert camera.k.dtype == torch.float32 and camera.size == (40, 24)
    for args, match in [
        (("", k, c2w, (4, 4)), "name"),
        (("a", torch.eye(4), c2w, (4, 4)), "k must be 3x3"),
        (("a", k, torch.eye(3), (4, 4)), "c2w must be 4x4"),
        (("a", k * math.nan, c2w, (4, 4)), "k has nonfinite"),
        (("a", k, c2w, (4, 0)), "size"),
        (("a", k, c2w, (4.5, 4)), "size"),
    ]:
        with pytest.raises(ValueError, match=match):
            Camera(*args)


def test_read_transforms_json(tmp_path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no distortion: no warning
        cameras = read_cameras(_scene(tmp_path / "scene") / "transforms.json")
    # File order (the scene lists its frames backwards); file names only.
    assert [c.name for c in cameras] == [
        f"frame_{i:05d}.png" for i in reversed(range(8))
    ]
    assert all(c.size == (40, 24) for c in cameras)
    torch.testing.assert_close(cameras[0].k[0, 0], torch.tensor(50.0 / 40))  # per frame
    torch.testing.assert_close(cameras[1].k[0, 0], torch.tensor(30.0 / 40))
    torch.testing.assert_close(cameras[1].k[1, 1], torch.tensor(32.0 / 24))
    torch.testing.assert_close(cameras[1].k[:2, 2], torch.tensor([0.5, 0.5]))
    # OpenGL -> OpenCV: y and z axes flipped, position kept.
    c2w = cameras[4].c2w  # frame 3
    assert torch.equal(c2w[:3, :3], torch.diag(torch.tensor([1.0, -1.0, -1.0])))
    torch.testing.assert_close(c2w[:3, 3], torch.tensor([0.3, 0.0, 0.0]))


def test_read_transforms_json_variants(tmp_path) -> None:
    pose = torch.eye(4).tolist()
    meta = {
        "camera_angle_x": 2 * math.atan(0.5),  # fx = w
        "w": 40,
        "h": 20,
        "camera_model": "OPENCV",
        "k1": 0.1,
        "p2": 0.02,
        "frames": [
            {"file_path": "./images/a", "transform_matrix": pose},
            {
                "file_path": "b.jpg",
                "transform_matrix": pose,
                "camera_angle_y": 2 * math.atan(1.0),
                "cx": 10,
            },
            {
                "file_path": "c.png",
                "transform_matrix": pose,
                "camera_model": "OPENCV_FISHEYE",
            },
            {"file_path": "d.png", "transform_matrix": pose, "camera_model": "PINHOLE"},
        ],
    }
    path = _transforms(tmp_path / "t.json", meta)
    # Distortion is ignored, with one warning naming the models that have it.
    with pytest.warns(
        UserWarning,
        match="t.json: ignoring the lens distortion of its "
        "OPENCV, OPENCV_FISHEYE cameras",
    ) as caught:
        a, b, c, d = read_cameras(path)
    assert len(caught) == 1
    assert [x.name for x in (a, b, c, d)] == ["a.png", "b.jpg", "c.png", "d.png"]
    torch.testing.assert_close(a.k[0], torch.tensor([1.0, 0.0, 0.5]))
    torch.testing.assert_close(a.k[1], torch.tensor([0.0, 2.0, 0.5]))  # fy = fx
    torch.testing.assert_close(b.k[1, 1], torch.tensor(0.5))  # fy = h / 2
    torch.testing.assert_close(b.k[0, 2], torch.tensor(0.25))
    # OPENCV without distortion coefficients is a pinhole camera: no warning.
    meta.pop("k1"), meta.pop("p2"), meta["frames"].pop(2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        read_cameras(_transforms(tmp_path / "t.json", meta))


@pytest.mark.parametrize(
    "change, match",
    [
        (
            {"camera_model": "EQUIRECTANGULAR"},
            "frame 0 .* uses the EQUIRECTANGULAR camera model, which has no pinhole",
        ),
        ({"w": None}, "has no w"),
        ({"fl_x": None}, "fl_x or camera_angle_x"),
        ({"transform_matrix": [[1, 0], [0, 1]]}, "4x4"),
        ({"transform_matrix": [[math.nan] * 4] * 4}, "c2w has nonfinite"),
        ({"fl_x": "wide"}, "frame 0 \\(a.png\\)"),
    ],
)
def test_read_transforms_json_errors(tmp_path, change, match) -> None:
    frame = {"file_path": "a.png", "transform_matrix": torch.eye(4).tolist()}
    meta = {"fl_x": 10, "w": 20, "h": 10}
    for key, value in change.items():
        target = frame if key == "transform_matrix" else meta
        if value is None:
            target.pop(key)
        else:
            target[key] = value
    path = tmp_path / "t.json"
    path.write_text(json.dumps({**meta, "frames": [frame]}))
    with pytest.raises(ValueError, match=match):
        read_cameras(path)


@pytest.mark.parametrize("ext", ["txt", "bin"])
def test_read_colmap_matches_transforms(tmp_path, ext) -> None:
    cams = [
        (1, "PINHOLE", 40, 30, (30.0, 32.0, 19.0, 15.5)),
        (2, "SIMPLE_PINHOLE", 20, 10, (12.0, 10.0, 5.0)),
    ]
    images = [
        (i + 1, q, t, 1 + i % 2, f"sub/img_{i}.png")
        for i, (q, t) in enumerate(zip(QVECS, TVECS))
    ]
    _write_colmap(tmp_path / "sparse", cams, images, ext)
    colmap = read_cameras(tmp_path / "sparse")

    frames = []
    for i, (q, t) in enumerate(zip(QVECS, TVECS)):
        k = {"fl_x": 30.0, "fl_y": 32.0, "cx": 19.0, "cy": 15.5, "w": 40, "h": 30}
        if i % 2:
            k = {"fl_x": 12.0, "cx": 10.0, "cy": 5.0, "w": 20, "h": 10}
        gl = _c2w(q, t) @ GL_TO_CV
        frames.append(
            {"file_path": f"images/img_{i}.png", "transform_matrix": gl.tolist(), **k}
        )
    transforms = read_cameras(_transforms(tmp_path / "t.json", {"frames": frames}))
    _assert_same(colmap, transforms)
    assert [c.name for c in colmap] == ["img_0.png", "img_1.png", "img_2.png"]
    # The files themselves, as a list, read the same.
    files = [tmp_path / "sparse" / f"{n}.{ext}" for n in ("cameras", "images")]
    _assert_same(read_cameras(files), colmap)


@pytest.mark.parametrize("ext", ["txt", "bin"])
@pytest.mark.parametrize(
    "model, params, k, distorted",
    [
        ("SIMPLE_PINHOLE", (10, 5, 4), (10, 10, 5, 4), False),
        ("PINHOLE", (10, 11, 5, 4), (10, 11, 5, 4), False),
        ("SIMPLE_RADIAL", (10, 5, 4, 0.1), (10, 10, 5, 4), True),
        ("RADIAL", (10, 5, 4, 0.1, 0.2), (10, 10, 5, 4), True),
        ("OPENCV", (10, 11, 5, 4, 0.1, 0.2, 0.01, 0.02), (10, 11, 5, 4), True),
        ("OPENCV", (10, 11, 5, 4, 0, 0, 0, 0), (10, 11, 5, 4), False),
        ("FULL_OPENCV", (10, 11, 5, 4, *range(1, 9)), (10, 11, 5, 4), True),
        ("OPENCV_FISHEYE", (10, 11, 5, 4, 0.1, 0.2, 0.3, 0.4), (10, 11, 5, 4), True),
        ("OPENCV_FISHEYE", (10, 11, 5, 4, 0, 0, 0, 0), (10, 11, 5, 4), True),
        ("FOV", (10, 11, 5, 4, 0.1), (10, 11, 5, 4), True),
        ("SIMPLE_RADIAL_FISHEYE", (10, 5, 4, 0.1), (10, 10, 5, 4), True),
        ("RADIAL_FISHEYE", (10, 5, 4, 0.1, 0.2), (10, 10, 5, 4), True),
        ("THIN_PRISM_FISHEYE", (10, 11, 5, 4, *[0.1] * 8), (10, 11, 5, 4), True),
        (
            "RAD_TAN_THIN_PRISM_FISHEYE",
            (10, 11, 5, 4, *[0.1] * 12),
            (10, 11, 5, 4),
            True,
        ),
    ],
)
def test_read_colmap_models(tmp_path, ext, model, params, k, distorted) -> None:
    _write_colmap(
        tmp_path,
        [(3, model, 20, 10, params)],
        [(1, QVECS[0], TVECS[0], 3, "a.png")],
        ext,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (camera,) = read_cameras(tmp_path)
    messages = [str(w.message) for w in caught]
    assert messages == (
        [
            f"{tmp_path}/{{cameras,images}}.{ext}: ignoring the lens distortion of its "
            f"{model} cameras (ffgs reads their pinhole intrinsics only)"
        ]
        if distorted
        else []
    )
    assert camera.size == (20, 10)
    fx, fy, cx, cy = k
    torch.testing.assert_close(
        camera.k,
        torch.tensor([[fx / 20, 0, cx / 20], [0, fy / 10, cy / 10], [0, 0, 1]]),
    )


@pytest.mark.parametrize("ext", ["txt", "bin"])
def test_read_colmap_errors(tmp_path, ext) -> None:
    image = [(1, QVECS[0], TVECS[0], 1, "a.png")]
    _write_colmap(tmp_path / "id", [(2, "PINHOLE", 20, 10, (10, 10, 5, 4))], image, ext)
    with pytest.raises(ValueError, match="image a.png has camera 1"):
        read_cameras(tmp_path / "id")
    # An unknown camera model (a name in txt, an id in bin).
    _write_colmap(tmp_path / "m", [(1, "PINHOLE", 20, 10, (10, 10, 5, 4))], image, ext)
    cameras = tmp_path / "m" / f"cameras.{ext}"
    if ext == "txt":
        cameras.write_text(cameras.read_text().replace("PINHOLE", "WEIRD"))
        match = r"camera 1 \(image a.png\) has the unknown COLMAP camera model WEIRD"
    else:
        data = bytearray(cameras.read_bytes())
        data[12:16] = struct.pack("<i", 99)
        cameras.write_bytes(bytes(data))
        match = "unknown COLMAP camera model id 99"
    with pytest.raises(ValueError, match=match):
        read_cameras(tmp_path / "m")
    bad_q = [(1, (0.0, 0.0, 0.0, 0.0), TVECS[0], 1, "a.png")]
    _write_colmap(tmp_path / "q", [(1, "PINHOLE", 20, 10, (10, 10, 5, 4))], bad_q, ext)
    with pytest.raises(ValueError, match="invalid quaternion"):
        read_cameras(tmp_path / "q")
    broken = tmp_path / "q" / f"images.{ext}"
    broken.write_bytes(broken.read_bytes()[:12] if ext == "bin" else b"1 x y\n")
    with pytest.raises(ValueError, match="not a valid COLMAP model"):
        read_cameras(tmp_path / "q")


def test_save_cameras_round_trip(tmp_path) -> None:
    k = torch.tensor([[0.8, 0, 0.5], [0, 1.1, 0.45], [0, 0, 1]])
    c2w = _c2w(QVECS[1], TVECS[1]).float()
    cameras = [
        Camera("a.png", k, c2w, (40, 24)),
        Camera("b.png", k * 0.5, torch.eye(4), (640, 480)),
    ]
    path = save_cameras(tmp_path / "cameras.json", cameras)
    data = json.loads(path.read_text())
    assert (
        isinstance(data, list)
        and [set(entry) for entry in data] == [{"name", "k", "c2w", "size"}] * 2
    )
    assert data[1]["size"] == [640, 480]
    again = read_cameras(path)
    assert [(c.name, c.size) for c in again] == [(c.name, c.size) for c in cameras]
    for x, y in zip(again, cameras):  # exactly
        assert torch.equal(x.k, y.k) and torch.equal(x.c2w, y.c2w)
    save_cameras(tmp_path / "again.json", again)
    assert (tmp_path / "again.json").read_text() == path.read_text()
    with pytest.raises(TypeError, match="Camera objects"):
        save_cameras(tmp_path / "x.json", [{"name": "a"}])


def test_read_ffgs_cameras_json_errors(tmp_path) -> None:
    entry = {
        "name": "a.png",
        "k": torch.eye(3).tolist(),
        "c2w": torch.eye(4).tolist(),
        "size": [4, 3],
    }
    path = tmp_path / "cameras.json"
    path.write_text(json.dumps([entry, {"k": entry["k"]}]))
    with pytest.raises(ValueError, match="camera 1 has no name, c2w, size"):
        read_cameras(path)
    path.write_text(json.dumps([{**entry, "size": [4, 0]}]))
    with pytest.raises(ValueError, match="cameras.json: camera a.png: size"):
        read_cameras(path)
    path.write_text(json.dumps([entry, 3]))
    with pytest.raises(ValueError, match="camera 1 is not an object"):
        read_cameras(path)


def test_read_cameras_unknown_formats(tmp_path) -> None:
    (tmp_path / "views.json").write_text('{"views": []}')
    (tmp_path / "broken.json").write_text("{")
    (tmp_path / "poses.txt").write_text("1 2 3")
    (tmp_path / "empty").mkdir()
    for path in ("views.json", "poses.txt", "empty"):
        with pytest.raises(
            ValueError, match="not a camera file ffgs reads.*build ffgs.io.Camera"
        ):
            read_cameras(tmp_path / path)
    with pytest.raises(ValueError, match="not a camera file"):
        read_cameras([tmp_path / "poses.txt", tmp_path / "views.json"])
    with pytest.raises(ValueError, match="not valid JSON"):
        read_cameras(tmp_path / "broken.json")
    with pytest.raises(FileNotFoundError):
        read_cameras(tmp_path / "missing.json")


# --- pairing and images -------------------------------------------------------------


def _named(*names: str) -> list[Camera]:
    cameras = []
    for i, name in enumerate(names):
        c2w = torch.eye(4)
        c2w[0, 3] = float(i)  # camera i at x = i
        cameras.append(Camera(name, torch.eye(3), c2w, (40, 24)))
    return cameras


def test_pair_cameras_by_file_name(tmp_path) -> None:
    cameras = _named("b.png", "a.png", "c.png")
    paths = [tmp_path / "a.png", Path("x/c.png"), Path("b.png")]
    assert [c.name for c in pair_cameras(paths, cameras)] == ["a.png", "c.png", "b.png"]
    with pytest.raises(ValueError, match="no camera for 1 image.*: d.png"):
        pair_cameras(paths + [Path("d.png")], cameras)
    with pytest.raises(ValueError, match="two cameras are named a.png"):
        pair_cameras(paths, cameras + cameras[1:2])
    with pytest.raises(ValueError, match="two images are named a.png"):
        pair_cameras([Path("a.png"), Path("y/a.png")], cameras)
    with pytest.raises(ValueError, match="no cameras"):
        pair_cameras(paths, [])


def test_pair_cameras_by_position() -> None:
    cameras = _named("0", "1", "2")
    paths = [Path("z.png"), Path("y.png"), Path("x.png")]
    assert pair_cameras(paths, cameras) == cameras  # in order, whatever the names
    with pytest.raises(ValueError, match="3 cameras named by position, but 2 images"):
        pair_cameras(paths[:2], cameras)


def test_image_helpers(tmp_path) -> None:
    folder = tmp_path / "images"
    folder.mkdir()
    for name in ("b.png", "a.JPG", "c.jpeg"):
        Image.new("RGB", (4, 3)).save(folder / name, format="PNG")
    (folder / "notes.txt").write_text("not an image")
    assert [p.name for p in io.image_paths(folder)] == ["a.JPG", "b.png", "c.jpeg"]
    files = [folder / "c.jpeg", folder / "a.JPG"]
    assert io.image_paths(files) == files  # in the order given
    image = io.read_image(folder / "b.png")
    assert image.shape == (3, 3, 4) and image.dtype == torch.uint8
    assert io.evenly_spaced(6, 3) == [0, 2, 5] and io.evenly_spaced(2, 5) == [0, 1]
    with pytest.raises(FileNotFoundError, match="not a directory"):
        io.image_paths(tmp_path / "nothing")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no .png"):
        io.image_paths(tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="not a file"):
        io.image_paths([tmp_path / "nothing.png"])
    (tmp_path / "broken.png").write_bytes(b"not a png")
    with pytest.raises(ValueError, match="cannot read the image .*broken.png"):
        io.read_image(tmp_path / "broken.png")
