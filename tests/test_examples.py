"""`examples/` scripts run end to end on a synthetic scene with the dummy model."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from _dummy import (
    DummyModel,
    DummyProcessor,
    PoseFreeDummyModel,
    PoseFreeDummyProcessor,
)
from PIL import Image

from ffgs import GSPipeline
from ffgs.io import read_cameras

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene(root: Path, n: int = 8, h: int = 24, w: int = 40) -> Path:
    """nerfstudio layout; frames listed out of order, the last with its own K."""
    (root / "images").mkdir(parents=True)
    frames = []
    for i in reversed(range(n)):
        name = f"images/frame_{i:05d}.png"
        pixels = np.random.default_rng(i).integers(0, 256, (h, w, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / name)
        c2w = torch.eye(4)
        c2w[:3, 3] = torch.tensor([0.1 * i, 0.0, 0.0])
        frames.append({"file_path": name, "transform_matrix": c2w.tolist()})
    frames[0].update(fl_x=50.0, fl_y=50.0)
    meta = {"fl_x": 30.0, "fl_y": 32.0, "cx": 20.0, "cy": 12.0, "w": w, "h": h}
    (root / "transforms.json").write_text(json.dumps({**meta, "frames": frames}))
    return root


@pytest.mark.parametrize("render", [False, pytest.param(True, marks=pytest.mark.gpu)])
def test_infer_nerfstudio_scene(tmp_path, monkeypatch, render) -> None:
    example = _load("infer_nerfstudio_scene")
    scene = _scene(tmp_path / "scene")
    model_dir = GSPipeline(DummyModel(), DummyProcessor()).save_pretrained(
        tmp_path / "model"
    )
    out = tmp_path / "out"
    argv = ["infer", str(scene), "--model", str(model_dir), "--out", str(out)]
    argv += ["--num-views", "3", "--num-render", "2" if render else "0"]
    monkeypatch.setattr(sys, "argv", argv)
    outputs = []
    call = GSPipeline.__call__

    def recording_call(self, *args, **kwargs):
        outputs.append(call(self, *args, **kwargs))
        return outputs[-1]

    monkeypatch.setattr(GSPipeline, "__call__", recording_call)

    example.main()

    # The ply holds the Gaussians in the scene's frame, not the model's.
    header, data = (out / "gaussians.ply").read_bytes().split(b"end_header\n")
    columns = header.count(b"property float")
    xyz = np.frombuffer(data, dtype="<f4").reshape(-1, columns)[:, :3]
    (output,) = outputs
    world = output.to_world().gaussians.means[0].cpu().numpy()
    np.testing.assert_array_equal(xyz, world)
    assert not np.allclose(xyz, output.gaussians.means[0].cpu().numpy())
    renders = sorted(p.name for p in out.glob("render_*.png"))
    # Inputs 0, 4, 7 (of 0..7): frames 2 and 5 lie between them.
    assert renders == (["render_00002.png", "render_00005.png"] if render else [])


def test_infer_nerfstudio_scene_one_view(tmp_path, monkeypatch, capsys) -> None:
    example = _load("infer_nerfstudio_scene")
    scene = _scene(tmp_path / "scene")
    model_dir = GSPipeline(DummyModel(), DummyProcessor()).save_pretrained(
        tmp_path / "model"
    )
    out = tmp_path / "out"
    argv = ["infer", str(scene), "--model", str(model_dir), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv + ["--num-views", "1"])
    example.main()
    assert (out / "gaussians.ply").exists() and not list(out.glob("render_*.png"))
    assert "no frames between inputs" in capsys.readouterr().out


def _image_folder(root: Path, n: int = 6, h: int = 24, w: int = 40) -> Path:
    root.mkdir(parents=True)
    for i in reversed(range(n)):
        pixels = np.random.default_rng(i).integers(0, 256, (h, w, 3), dtype=np.uint8)
        Image.fromarray(pixels).save(root / f"img_{i:03d}.png")
    (root / "notes.txt").write_text("not an image")
    return root


@pytest.mark.parametrize("render", [False, pytest.param(True, marks=pytest.mark.gpu)])
def test_infer_pose_free(tmp_path, monkeypatch, render) -> None:
    example = _load("infer_pose_free")
    folder = _image_folder(tmp_path / "images")
    torch.manual_seed(0)
    pipe = GSPipeline(PoseFreeDummyModel(), PoseFreeDummyProcessor())
    model_dir = pipe.save_pretrained(tmp_path / "model")
    out = tmp_path / "out"
    argv = ["infer", str(folder), "--model", str(model_dir), "--out", str(out)]
    argv += ["--num-views", "3"]
    argv += ["--device", "cuda"] if render else ["--device", "cpu", "--no-render"]
    monkeypatch.setattr(sys, "argv", argv)
    outputs = []
    call = GSPipeline.__call__

    def recording_call(self, views, *args, **kwargs):
        assert views.intrinsics is None and views.c2w is None
        outputs.append(call(self, views, *args, **kwargs))
        return outputs[-1]

    monkeypatch.setattr(GSPipeline, "__call__", recording_call)

    example.main()

    (output,) = outputs
    assert output.frame is None
    # The ply holds the Gaussians in the model's frame (there is no other).
    header, data = (out / "gaussians.ply").read_bytes().split(b"end_header\n")
    columns = header.count(b"property float")
    xyz = np.frombuffer(data, dtype="<f4").reshape(-1, columns)[:, :3]
    np.testing.assert_array_equal(xyz, output.gaussians.means[0].cpu().numpy())
    # The predicted cameras as they are, named by position (inputs: images 0, 2
    # and 5 of 0..5, evenly spaced by file name).
    cameras = read_cameras(out / "cameras.json")
    assert [c.name for c in cameras] == ["0", "1", "2"]
    h, w = output.cameras.image_shape
    assert all(c.size == (w, h) for c in cameras)
    c2w = torch.stack([c.c2w for c in cameras])
    torch.testing.assert_close(c2w, output.cameras.c2w[0].cpu())
    k = torch.stack([c.k for c in cameras])
    torch.testing.assert_close(k, output.cameras.normalized_k[0].cpu())
    renders = sorted(p.name for p in out.glob("render_*.png"))
    expected = ["render_0.png", "render_1.png", "render_2.png"]
    assert renders == (expected if render else [])
