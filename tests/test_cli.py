"""The `ffgs` command, called as `main([...])` over fake Hub caches and the dummy
CPU models."""

import json

import pytest
import torch
from _dummy import (
    DummyModel,
    DummyProcessor,
    PoseFreeDummyModel,
    PoseFreeDummyProcessor,
)
from _hub_cache import COMMIT, REPO, _cached, _ref
from PIL import Image
from test_examples import _image_folder, _scene
from test_io import QVECS, TVECS, _write_colmap

from ffgs import GSPipeline, __version__, read_ply_meta, store
from ffgs.cli import main
from ffgs.io import Camera, read_cameras, save_cameras


@pytest.fixture
def posed(tmp_path):
    return GSPipeline(DummyModel(), DummyProcessor()).save_pretrained(
        tmp_path / "posed"
    )


@pytest.fixture
def pose_free(tmp_path):
    torch.manual_seed(0)
    pipe = GSPipeline(PoseFreeDummyModel(), PoseFreeDummyProcessor())
    return pipe.save_pretrained(tmp_path / "pose-free")


def test_version_and_usage_errors(capsys) -> None:
    with pytest.raises(SystemExit) as exit:
        main(["--version"])
    assert exit.value.code == 0 and __version__ in capsys.readouterr().out
    for argv in ([], ["run", "x"], ["run", "a", "b", "-o", "c", "--num-views", "0"]):
        with pytest.raises(SystemExit) as exit:
            main(argv)
        assert exit.value.code == 2


def test_add_ls_rm(entries, cache, capsys) -> None:
    assert main(["ls"]) == 0
    assert "no models recorded" in capsys.readouterr().out
    assert main(["add", "dummy/v1"]) == 0
    assert main(["add", "someone/repo", "--revision", "v2"]) == 0
    assert "added dummy/v1 (not cached)" in capsys.readouterr().out
    _cached(cache, REPO, COMMIT, "ckpts/v1.pt", entries["v1"])

    assert main(["ls"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split()[:3] == ["SOURCE", "TYPE", "REVISION"]
    assert lines[1].split()[:4] == ["dummy/v1", "zoo", "-", "cached"]
    assert lines[2].split()[:4] == ["someone/repo", "hub", "v2", "not"]
    assert main(["ls", "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [(r["source"], r["status"]) for r in rows] == [
        ("dummy/v1", "cached"),
        ("someone/repo", "not cached"),
    ]

    assert main(["rm", "dummy/v1", "someone/repo"]) == 0
    assert "removed someone/repo@v2" in capsys.readouterr().out
    assert store.records() == []
    assert main(["rm", "dummy/v1"]) == 1
    assert "has no record" in capsys.readouterr().err


def test_rm_a_deleted_local_directory(posed, capsys) -> None:
    main(["add", str(posed)])
    for path in sorted(posed.rglob("*"), reverse=True):
        path.rmdir() if path.is_dir() else path.unlink()
    posed.rmdir()
    main(["ls"])
    assert "missing" in capsys.readouterr().out
    assert main(["rm", str(posed)]) == 0 and store.records() == []


def test_ls_zoo(entries, cache, capsys) -> None:
    store.add("dummy/v2")
    _cached(cache, REPO, COMMIT, "ckpts/v1.pt", entries["v1"])
    assert main(["ls", "--zoo", "--json"]) == 0
    rows = {r["name"]: r for r in json.loads(capsys.readouterr().out)}
    assert rows["dummy/v1"]["status"] == "cached" and not rows["dummy/v1"]["recorded"]
    assert rows["dummy/v2"]["status"] == "not cached" and rows["dummy/v2"]["recorded"]
    assert rows["dummy/v1"]["license"] == "MIT"
    assert main(["ls", "--zoo"]) == 0
    assert capsys.readouterr().out.splitlines()[0].startswith("NAME")


def test_pull_errors_are_one_line_unless_debug(entries, cache, capsys) -> None:
    assert main(["pull", "dummy/v1", "--revision", "main"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("ffgs: error:") and "pinned" in err
    assert len(err.splitlines()) == 1
    assert main(["pull", "dummy/v1", "--revision", "main", "--debug"]) == 1
    assert "Traceback" in capsys.readouterr().err


def test_run_posed(posed, tmp_path, capsys) -> None:
    scene = _scene(tmp_path / "scene")
    out = tmp_path / "out"
    (out).mkdir()
    (out / "keep.txt").write_text("not ours")
    argv = ["run", str(posed), str(scene / "images"), "-o", str(out)]
    argv += ["--cameras", str(scene / "transforms.json"), "--num-views", "3"]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert "(world frame)" in printed and "cameras" not in printed

    assert read_ply_meta(out / "scene.ply")["space"] == "world"
    assert not (out / "cameras.json").exists()  # the model predicts none
    assert (out / "keep.txt").exists()
    (record,) = store.records()  # recorded by running it
    assert record.source == str(posed) and record.last_used is not None


def test_run_needs_cameras(posed, tmp_path, capsys) -> None:
    folder = _image_folder(tmp_path / "images")
    out = tmp_path / "out"
    assert main(["run", str(posed), str(folder), "-o", str(out)]) == 1
    assert "--cameras" in capsys.readouterr().err
    assert not out.exists() and store.records() == []  # failed before loading


def test_run_pose_free(pose_free, tmp_path, capsys) -> None:
    folder = _image_folder(tmp_path / "images")
    out = tmp_path / "out"
    argv = ["run", str(pose_free), str(folder), "-o", str(out), "--device", "cpu"]
    assert main(argv) == 0
    printed = capsys.readouterr().out
    assert "6 predicted cameras (model frame)" in printed  # all 6 (default 8)
    assert read_ply_meta(out / "scene.ply")["space"] == "model"
    cameras = read_cameras(out / "cameras.json")
    assert [c.name for c in cameras] == [str(i) for i in range(6)]  # by position
    assert all(c.size == (32, 16) for c in cameras)


@pytest.mark.parametrize("ext", ["txt", "bin"])
def test_run_with_colmap_cameras(pose_free, tmp_path, capsys, ext) -> None:
    folder = _image_folder(tmp_path / "images")
    names = sorted(p.name for p in folder.glob("*.png"))
    images = [(i + 1, QVECS[i % 3], TVECS[i % 3], 1, n) for i, n in enumerate(names)]
    _write_colmap(
        tmp_path / "sparse", [(1, "PINHOLE", 40, 24, (30, 30, 20, 12))], images, ext
    )
    out = tmp_path / "out"
    argv = ["run", str(pose_free), str(folder), "-o", str(out), "--device", "cpu"]
    assert main(argv + ["--cameras", str(tmp_path / "sparse")]) == 0
    assert "6 predicted cameras (world frame)" in capsys.readouterr().out
    assert read_ply_meta(out / "scene.ply")["space"] == "world"
    assert len(read_cameras(out / "cameras.json")) == 6


def test_run_reads_views(posed, tmp_path, monkeypatch) -> None:
    scene = _scene(tmp_path / "scene")
    images, transforms = scene / "images", scene / "transforms.json"
    seen = []
    call = GSPipeline.__call__

    def recording_call(self, views, *args, **kwargs):
        seen.append(views)
        return call(self, views, *args, **kwargs)

    monkeypatch.setattr(GSPipeline, "__call__", recording_call)
    argv = ["run", str(posed), str(images), "-o", str(tmp_path / "out")]
    argv += ["--num-views", "3"]
    # Evenly spaced images, each with the camera of its file name.
    assert main(argv + ["--cameras", str(transforms)]) == 0
    cameras = {c.name: c for c in read_cameras(transforms)}
    names = ["frame_00000.png", "frame_00004.png", "frame_00007.png"]
    views = seen.pop()
    assert views.normalized_intrinsics
    torch.testing.assert_close(
        views.c2w[0], torch.stack([cameras[n].c2w for n in names])
    )
    torch.testing.assert_close(
        views.intrinsics[0], torch.stack([cameras[n].k for n in names])
    )
    # Cameras named by position pair in order: the i-th for the i-th image.
    by_position = [
        Camera(str(i), cameras[f"frame_{i:05d}.png"].k, torch.eye(4), (40, 24))
        for i in range(8)
    ]
    for i, camera in enumerate(by_position):
        camera.c2w[0, 3] = float(i)
    path = save_cameras(tmp_path / "cameras.json", by_position)
    assert main(argv + ["--cameras", str(path)]) == 0
    assert seen.pop().c2w[0, :, 0, 3].tolist() == [0.0, 4.0, 7.0]
    # A half-size copy keeps the aspect ratio: its cameras still hold.
    small = tmp_path / "small"
    small.mkdir()
    for path in sorted(images.iterdir()):
        with Image.open(path) as image:
            image.resize((20, 12)).save(small / path.name)
    argv[2] = str(small)
    assert main(argv + ["--cameras", str(transforms)]) == 0
    assert seen.pop().images.shape[-2:] == (12, 20)


def test_run_view_errors(posed, tmp_path, capsys) -> None:
    scene = _scene(tmp_path / "scene")
    images, transforms = scene / "images", scene / "transforms.json"
    argv = ["run", str(posed), str(images), "-o", str(tmp_path / "out")]
    argv += ["--cameras", str(transforms)]

    def error() -> str:
        assert main(argv) == 1
        return capsys.readouterr().err

    Image.new("RGB", (40, 24)).save(images / "zzz.png")
    assert "no camera for 1 image(s): zzz.png" in error()
    (images / "zzz.png").unlink()
    Image.new("RGB", (24, 24)).save(images / "frame_00000.png")
    assert "not all the same size" in error()
    for i in range(8):
        Image.new("RGB", (24, 24)).save(images / f"frame_{i:05d}.png")
    assert "a different aspect ratio" in error()


def test_run_without_cameras_in_or_out(tmp_path, clean_registry) -> None:
    from _dummy import CameraFreeDummyProcessor

    from ffgs import ModelSpec, register_model

    spec = ModelSpec(
        "ffgs-test-camera-free", PoseFreeDummyModel, CameraFreeDummyProcessor
    )
    register_model(spec)
    pipe = GSPipeline(PoseFreeDummyModel(), CameraFreeDummyProcessor(), spec.model_type)
    model = pipe.save_pretrained(tmp_path / "model")
    folder = _image_folder(tmp_path / "images")
    out = tmp_path / "out"
    assert main(["run", str(model), str(folder), "-o", str(out)]) == 0
    assert (out / "scene.ply").exists() and not (out / "cameras.json").exists()


def test_run_downloads_again_what_was_deleted(
    posed, cache, tmp_path, monkeypatch
) -> None:
    import huggingface_hub

    downloads = []

    def hf_hub_download(repo_id, filename, subfolder=None, revision=None, **kwargs):
        name = f"{subfolder}/{filename}" if subfolder else filename
        link = cache / f"models--{repo_id.replace('/', '--')}" / "snapshots"
        link = link / COMMIT / name
        if not link.exists():
            downloads.append(name)
            _cached(cache, repo_id, COMMIT, name, (posed / name).read_bytes())
            _ref(cache, repo_id, "main", COMMIT)
        return str(link)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(huggingface_hub.hub_mixin, "hf_hub_download", hf_hub_download)
    scene = _scene(tmp_path / "scene")
    argv = ["run", "someone/dummy", str(scene / "images"), "-o", str(tmp_path / "o")]
    argv += ["--cameras", str(scene / "transforms.json")]
    assert main(argv) == 0
    record = store.get("someone/dummy")
    assert record.commit == COMMIT and store.status(record).state == "cached"
    assert "model.safetensors" in downloads

    for path in store._files(record):  # deleted behind ffgs's back (`hf cache rm`)
        path.unlink()
    assert store.status(store.get("someone/dummy")).state == "not cached"
    downloads.clear()
    assert main(argv) == 0
    assert "model.safetensors" in downloads
    assert store.status(store.get("someone/dummy")).state == "cached"
