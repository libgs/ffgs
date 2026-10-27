"""Camera files, and the image helpers that go with them.

A camera file holds a list of `Camera`s, in order (each: `name`, normalised `k`,
OpenCV `c2w`, `size`). `read_cameras` reads them from a transforms.json, a COLMAP
model or an ffgs cameras.json; `save_cameras` writes them as an ffgs
cameras.json, which `read_cameras` reads back as they were. `pair_cameras` gives
each image its camera.

This module only reads and writes files: it knows nothing of `Views` or
`Cameras`. Building those (and checking the images fit the cameras) is the
caller's.
"""

from __future__ import annotations

import json
import math
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
# OpenGL / Blender camera axes (y up, z backward) -> OpenCV (y down, z forward).
GL_TO_CV = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64))


def image_paths(images: str | Path | list[str | Path]) -> list[Path]:
    """The images of a folder (.png / .jpg / .jpeg, sorted by file name), or the
    image files `images`, in their order."""
    if isinstance(images, (list, tuple)):
        paths = [Path(p) for p in images]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"{path} is not a file")
        if not paths:
            raise FileNotFoundError("no images given")
        return paths
    folder = Path(images)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a directory")
    paths = sorted(
        (p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )
    if not paths:
        raise FileNotFoundError(f"no {'/'.join(IMAGE_SUFFIXES)} images in {folder}")
    return paths


def read_image(path: str | Path) -> torch.Tensor:
    """uint8 [3, H, W]."""
    try:
        with Image.open(path) as image:
            array = np.array(image.convert("RGB"))
    except OSError as error:
        raise ValueError(f"cannot read the image {path}: {error}") from error
    return torch.from_numpy(array).permute(2, 0, 1)


def evenly_spaced(n: int, count: int) -> list[int]:
    """`count` indices of `range(n)` spread evenly, both ends included (all of
    them when `count >= n`)."""
    if n < 1 or count < 1:
        raise ValueError(f"need n >= 1 and count >= 1, got {n} and {count}")
    return torch.linspace(0, n - 1, min(count, n)).round().long().unique().tolist()


# --- cameras ------------------------------------------------------------------------

CAMERA_FORMATS = (
    "a nerfstudio / instant-ngp transforms.json",
    "a COLMAP model (cameras + images, .bin or .txt)",
    "an ffgs cameras.json",
)


@dataclass(frozen=True, eq=False)
class Camera:
    """One pinhole camera: what a camera file says about one image.

    `k` is normalised by `size` (fx / w, cx / w, fy / h, cy / h), `c2w` has OpenCV
    axes, `size` is the (w, h) the camera is for. `name` is the image's file
    name, or the camera's position when its source names none ("0", "1", ...).
    """

    name: str
    k: torch.Tensor
    c2w: torch.Tensor
    size: tuple[int, int]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"a camera's name must be a nonempty str: {self.name!r}")
        for field, shape in (("k", (3, 3)), ("c2w", (4, 4))):
            value = torch.as_tensor(getattr(self, field), dtype=torch.float32)
            if value.shape != shape:
                raise ValueError(
                    f"camera {self.name}: {field} must be {shape[0]}x{shape[1]}, "
                    f"got {tuple(value.shape)}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"camera {self.name}: {field} has nonfinite values")
            object.__setattr__(self, field, value)
        size = tuple(self.size)
        if len(size) != 2 or any(int(x) != x or x < 1 for x in size):
            raise ValueError(
                f"camera {self.name}: size must be (w, h), two positive integers, "
                f"got {self.size!r}"
            )
        object.__setattr__(self, "size", (int(size[0]), int(size[1])))


def read_cameras(path: str | Path | list[str | Path]) -> list[Camera]:
    """The cameras of a camera file, in its order: a transforms.json, a COLMAP
    model or an ffgs cameras.json (see `save_cameras`). `path` is a .json file, a
    COLMAP model directory, or the files themselves (a .json, or COLMAP's
    `cameras` + `images`).

    Lens distortion is ignored, with a warning: a camera is its pinhole `k`.
    A camera without one (e.g. equirectangular) is an error."""
    if isinstance(path, (list, tuple)):
        files = [Path(p) for p in path]
        if len(files) == 1:
            return read_cameras(files[0])
        return _read_colmap(files, "the camera files")
    path = Path(path)
    if path.is_dir():
        return _read_colmap(list(path.iterdir()), str(path))
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path} is not valid JSON ({error})") from error
        if isinstance(data, list):
            return _read_ffgs(data, path)
        if isinstance(data, dict) and isinstance(data.get("frames"), list):
            return _read_transforms(data, path)
    raise _unknown_format(str(path))


def save_cameras(path: str | Path, cameras: list[Camera]) -> Path:
    """Write `cameras` to `path` as an ffgs cameras.json, as they are; the path.
    The file is a list, one entry per camera, in order:

        [{"name": "0001.png", "k": [[3x3]], "c2w": [[4x4]], "size": [w, h]}, ...]

    `read_cameras` reads it back to the same cameras."""
    entries = []
    for camera in cameras:
        if not isinstance(camera, Camera):
            raise TypeError(f"save_cameras takes Camera objects, got {type(camera)}")
        entries.append(
            {
                "name": camera.name,
                "k": camera.k.tolist(),
                "c2w": camera.c2w.tolist(),
                "size": list(camera.size),
            }
        )
    path = Path(path)
    path.write_text(json.dumps(entries, indent=1) + "\n")
    return path


def pair_cameras(paths: list[str | Path], cameras: list[Camera]) -> list[Camera]:
    """The camera of each image of `paths`: the one named by its file name, or,
    when the cameras are named by position ("0", "1", ...), the i-th camera for
    the i-th image (as many cameras as images)."""
    paths = [Path(p) for p in paths]
    if not cameras:
        raise ValueError("there are no cameras")
    if [camera.name for camera in cameras] == [str(i) for i in range(len(cameras))]:
        if len(cameras) != len(paths):
            raise ValueError(
                f"{len(cameras)} cameras named by position, but {len(paths)} images: "
                "they pair in order, one each"
            )
        return list(cameras)
    by_name: dict[str, Camera] = {}
    for camera in cameras:
        if camera.name in by_name:
            raise ValueError(f"two cameras are named {camera.name}")
        by_name[camera.name] = camera
    seen: set[str] = set()
    for path in paths:
        if path.name in seen:
            raise ValueError(
                f"two images are named {path.name}: cameras pair with images by file name"
            )
        seen.add(path.name)
    absent = [p.name for p in paths if p.name not in by_name]
    if absent:
        raise ValueError(
            f"no camera for {len(absent)} image(s): " + ", ".join(absent[:5])
        )
    return [by_name[p.name] for p in paths]


def _unknown_format(source: str) -> ValueError:
    return ValueError(
        f"{source}: not a camera file ffgs reads. It reads "
        + "; ".join(CAMERA_FORMATS)
        + ". Other formats: build ffgs.io.Camera objects (or write an ffgs "
        "cameras.json) yourself"
    )


def _no_pinhole(source: str, where: str, model: str) -> ValueError:
    return ValueError(
        f"{source}: {where} uses the {model} camera model, which has no pinhole "
        "intrinsics"
    )


def _warn_distortion(source: str, models: set[str]) -> None:
    if models:
        warnings.warn(
            f"{source}: ignoring the lens distortion of its {', '.join(sorted(models))} "
            "cameras (ffgs reads their pinhole intrinsics only)",
            stacklevel=3,
        )


def _camera(source: str, **fields) -> Camera:
    try:
        return Camera(**fields)
    except (TypeError, ValueError) as error:  # it names the camera
        raise ValueError(f"{source}: {error}") from error


def _normalized_k(fx, fy, cx, cy, w, h) -> list[list[float]]:
    return [[fx / w, 0.0, cx / w], [0.0, fy / h, cy / h], [0.0, 0.0, 1.0]]


def _read_ffgs(entries: list, path: Path) -> list[Camera]:
    cameras = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: camera {i} is not an object")
        missing = [key for key in ("name", "k", "c2w", "size") if key not in entry]
        if missing:
            raise ValueError(f"{path}: camera {i} has no {', '.join(missing)}")
        fields = {key: entry[key] for key in ("name", "k", "c2w", "size")}
        cameras.append(_camera(str(path), **fields))
    return cameras


# nerfstudio's `camera_model`s with pinhole intrinsics (fl_x, fl_y, cx, cy); the
# others (EQUIRECTANGULAR, ...) have none.
TRANSFORMS_MODELS = (
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "OPENCV_FISHEYE",
    "FISHEYE624",
)
TRANSFORMS_DISTORTION = ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")


def _read_transforms(meta: dict, path: Path) -> list[Camera]:
    """nerfstudio / instant-ngp: per-frame values override the global ones;
    OpenGL `transform_matrix`."""
    cameras, distorted = [], set()
    for i, frame in enumerate(meta["frames"]):
        where = f"frame {i}"
        if not isinstance(frame, dict):
            raise ValueError(f"{path}: {where} is not an object")
        cam = {**meta, **frame}
        if "file_path" in frame:
            where = f"frame {i} ({frame['file_path']})"
        camera_model = cam.get("camera_model", "OPENCV")  # nerfstudio's default
        if camera_model not in TRANSFORMS_MODELS:
            raise _no_pinhole(str(path), where, camera_model)
        missing = [
            key for key in ("file_path", "transform_matrix", "w", "h") if key not in cam
        ]
        if "fl_x" not in cam and "camera_angle_x" not in cam:
            missing.append("fl_x or camera_angle_x")
        if missing:
            raise ValueError(f"{path}: {where} has no {', '.join(missing)}")
        try:
            w, h = float(cam["w"]), float(cam["h"])
            if "fl_x" in cam:
                fx = float(cam["fl_x"])
            else:
                fx = 0.5 * w / math.tan(0.5 * float(cam["camera_angle_x"]))
            if "fl_y" in cam:
                fy = float(cam["fl_y"])
            elif "camera_angle_y" in cam:
                fy = 0.5 * h / math.tan(0.5 * float(cam["camera_angle_y"]))
            else:
                fy = fx
            cx, cy = float(cam.get("cx", w / 2)), float(cam.get("cy", h / 2))
            coeffs = [float(cam.get(key, 0.0)) for key in TRANSFORMS_DISTORTION]
            matrix = torch.tensor(cam["transform_matrix"], dtype=torch.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: {where}: {error}") from error
        if matrix.shape != (4, 4):
            raise ValueError(
                f"{path}: {where}: transform_matrix must be 4x4, got {tuple(matrix.shape)}"
            )
        if "PINHOLE" not in camera_model and ("FISHEYE" in camera_model or any(coeffs)):
            distorted.add(camera_model)
        name = PurePosixPath(str(frame["file_path"])).name
        if not PurePosixPath(name).suffix:  # nerfstudio may leave it out
            name += ".png"
        cameras.append(
            _camera(
                str(path),
                name=name,
                k=_normalized_k(fx, fy, cx, cy, w, h),
                c2w=matrix @ GL_TO_CV,
                size=(w, h),
            )
        )
    _warn_distortion(str(path), distorted)
    return cameras


# COLMAP camera models by id: name and parameter count. All start with their
# pinhole intrinsics: (f, cx, cy) for the SINGLE_FOCAL ones, else (fx, fy, cx, cy);
# the rest of the parameters are the lens distortion.
COLMAP_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
}
COLMAP_PARAMS = {name: count for name, count in COLMAP_MODELS.values()}
SINGLE_FOCAL = (
    "SIMPLE_PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "SIMPLE_RADIAL_FISHEYE",
    "RADIAL_FISHEYE",
)


def _read_colmap(files: list[Path], source: str) -> list[Camera]:
    by_name = {file.name: file for file in files}
    for ext in ("bin", "txt"):
        cameras, images = by_name.get(f"cameras.{ext}"), by_name.get(f"images.{ext}")
        if cameras is not None and images is not None:
            break
    else:
        raise _unknown_format(source)
    for file in (cameras, images):
        if not file.is_file():
            raise FileNotFoundError(f"{file} does not exist")
    source = str(cameras.parent / f"{{cameras,images}}.{ext}")
    parse = _colmap_bin if ext == "bin" else _colmap_txt
    try:
        calibrations, poses = parse(cameras.read_bytes(), images.read_bytes())
    except (
        ValueError,
        KeyError,
        IndexError,
        struct.error,
        UnicodeDecodeError,
    ) as error:
        raise ValueError(f"{source}: not a valid COLMAP model ({error!r})") from error

    out, distorted = [], set()
    for name, qvec, tvec, camera_id in poses:
        where = f"image {name}"
        if camera_id not in calibrations:
            raise ValueError(
                f"{source}: {where} has camera {camera_id}, which is not in it"
            )
        model, width, height, params = calibrations[camera_id]
        if model not in COLMAP_PARAMS:
            raise ValueError(
                f"{source}: camera {camera_id} ({where}) has the unknown COLMAP "
                f"camera model {model}"
            )
        if len(params) != COLMAP_PARAMS[model]:
            raise ValueError(
                f"{source}: camera {camera_id} ({model}) has {len(params)} parameters, "
                f"not {COLMAP_PARAMS[model]}"
            )
        if model in SINGLE_FOCAL:
            (f, cx, cy), rest = params[:3], params[3:]
            fx = fy = f
        else:
            (fx, fy, cx, cy), rest = params[:4], params[4:]
        if "FISHEYE" in model or any(rest):
            distorted.add(model)
        q = torch.tensor(qvec, dtype=torch.float64)
        if not torch.isfinite(q).all() or q.norm() < 1e-12:
            raise ValueError(
                f"{source}: {where} has an invalid quaternion {list(qvec)}"
            )
        rotation = _quat_to_matrix(q)  # world to camera
        c2w = torch.eye(4, dtype=torch.float64)
        c2w[:3, :3] = rotation.T
        c2w[:3, 3] = -rotation.T @ torch.tensor(tvec, dtype=torch.float64)
        out.append(
            _camera(
                source,
                name=PurePosixPath(name).name,
                k=_normalized_k(fx, fy, cx, cy, width, height),
                c2w=c2w,
                size=(width, height),
            )
        )
    _warn_distortion(source, distorted)
    return out


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """A quaternion (w, x, y, z, any norm) as a 3x3 rotation."""
    w, x, y, z = (q / q.norm()).tolist()
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float64,
    )


def _colmap_txt(cameras: bytes, images: bytes):
    calibrations = {}
    for line in cameras.decode().splitlines():
        fields = line.split()
        if fields and not fields[0].startswith("#"):
            # CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
            calibrations[int(fields[0])] = (
                fields[1],
                int(fields[2]),
                int(fields[3]),
                [float(x) for x in fields[4:]],
            )
    poses = []
    lines = iter(images.decode().splitlines())
    for line in lines:
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        # IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME, then a line of 2D points.
        values = [float(x) for x in fields[1:8]]
        poses.append((" ".join(fields[9:]), values[:4], values[4:], int(fields[8])))
        next(lines, None)
    return calibrations, poses


def _colmap_bin(cameras: bytes, images: bytes):
    calibrations = {}
    (count,) = struct.unpack_from("<Q", cameras, 0)
    offset = 8
    for _ in range(count):
        camera_id, model_id, width, height = struct.unpack_from(
            "<iiQQ", cameras, offset
        )
        offset += 24
        if model_id not in COLMAP_MODELS:
            raise ValueError(f"unknown COLMAP camera model id {model_id}")
        model, num_params = COLMAP_MODELS[model_id]
        params = struct.unpack_from(f"<{num_params}d", cameras, offset)
        offset += 8 * num_params
        calibrations[camera_id] = (model, width, height, list(params))
    poses = []
    (count,) = struct.unpack_from("<Q", images, 0)
    offset = 8
    for _ in range(count):
        values = struct.unpack_from("<i7di", images, offset)
        offset += 64
        end = images.index(b"\0", offset)
        name = images[offset:end].decode()
        offset = end + 1
        (num_points,) = struct.unpack_from("<Q", images, offset)
        offset += 8 + 24 * num_points
        poses.append((name, list(values[1:5]), list(values[5:8]), values[8]))
    return calibrations, poses
