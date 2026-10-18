"""C3G official-source CPU references and released checkpoint contract.

Tiny deterministic fp32 inference is compared with assert_f32_close (1e-5),
not bitwise equality. The fixture generator uses original C3G forward code.
Full released weights are not included; the GPU test renders the tiny model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from _empty import meta_parameters
from _numeric import assert_f32_close
from fixtures.make_c3g_reference import (
    CASES,
    DINO,
    STRIDE,
    TINY,
    fill_weights,
    preprocessing_source,
)
from safetensors import safe_open

import ffgs.pipeline
from ffgs import Cameras, GSPipeline, Views, get_model_spec, zoo
from ffgs.models.c3g import C3G, C3GProcessor, convert_state_dict
from ffgs.models.c3g.vggt.layers.vision_transformer import DinoVisionTransformer

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def reference():
    with safe_open(
        str(FIXTURES / "c3g_reference.safetensors"), framework="pt"
    ) as handle:
        assert handle.metadata()["upstream"] == "40b2cc9"
        return {name: handle.get_tensor(name) for name in handle.keys()}


@pytest.mark.parametrize("index", range(3))
def test_official_preprocessing(reference, index):
    prefix = f"process.{index}."
    source = preprocessing_source(index)
    intrinsics = torch.tensor([[0.8, 0, 0.5], [0, 1.2, 0.5], [0, 0, 1]])
    views = Views(
        source[None].repeat(2, 1, 1, 1),
        intrinsics[None].repeat(2, 1, 1),
        normalized_intrinsics=True,
    )
    processor = C3GProcessor()
    prepared = processor.preprocess(views, torch.device("cpu"))
    assert prepared.frame is None
    for image in prepared.model_input[0]:
        assert_f32_close(image[:, ::STRIDE, ::STRIDE], reference[prefix + "images"])
    fitted = processor.fit_views(views)
    digest = hashlib.sha256(
        (fitted.images[0, 0] * 255).round().to(torch.uint8).numpy().tobytes()
    ).digest()
    assert list(digest) == reference[prefix + "sha256"].tolist()
    assert_f32_close(fitted.normalized_k[0, 0], reference[prefix + "intrinsics"])


@pytest.mark.parametrize("case", CASES)
def test_official_cpu_forward(reference, case):
    model = C3G(**TINY)
    model.backbone.aggregator.patch_embed = DinoVisionTransformer(**DINO)
    fill_weights(model)
    model.eval()
    with torch.no_grad():
        raw = model(reference[f"{case}.images"])
    # The model's output is upstream's adapter output (xyzw, [B, N, 3, 1] SH).
    assert_f32_close(raw.harmonics, reference[f"{case}.colors"], msg="harmonics")
    quats = reference[f"{case}.quats"]  # wxyz
    assert_f32_close(raw.rotations, quats[..., [1, 2, 3, 0]], msg="rotations")
    output = C3GProcessor().postprocess(raw)
    for name in ("means", "scales", "quats", "colors", "opacities"):
        expected = reference[f"{case}.{name}"]
        if name == "colors":
            expected = expected.transpose(-1, -2)
        assert_f32_close(getattr(output, name), expected, msg=name)
    assert output.sh_degree == 0
    assert C3GProcessor().predicted_cameras(raw) is None


def test_released_checkpoint_schema():
    metadata = json.loads((FIXTURES / "c3g_checkpoint.json").read_text())
    entry = zoo.get_entry("c3g/default")
    assert metadata["sha256"] == entry.weights.sha256
    assert metadata["revision"] == entry.weights.revision
    assert metadata["size"] == 4624144072
    with meta_parameters():
        model = C3G()
    expected = model.state_dict()
    actual = {
        name.removeprefix("encoder."): info
        for name, info in metadata["tensors"].items()
    }
    assert len(actual) == len(expected) == 1300
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        assert list(tensor.shape) == actual[name]["shape"], name
        assert str(tensor.dtype) == actual[name]["dtype"], name
    state = {
        name: torch.empty(
            info["shape"],
            dtype=getattr(torch, info["dtype"].split(".")[1]),
            device="meta",
        )
        for name, info in metadata["tensors"].items()
    }
    model.load_state_dict(convert_state_dict({"state_dict": state}), strict=True)


def test_checkpoint_mapping_and_strictness():
    model = C3G(**TINY)
    checkpoint = {
        "state_dict": {
            f"encoder.{name}": value for name, value in model.state_dict().items()
        },
        "optimizer_states": [],
    }
    state = convert_state_dict(checkpoint)
    model.load_state_dict(state, strict=True)
    state.pop("gaussian_tokens")
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(state, strict=True)
    state["unexpected"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        model.load_state_dict(state, strict=True)
    with pytest.raises(ValueError, match="state_dict"):
        convert_state_dict({})
    with pytest.raises(ValueError, match="non-encoder"):
        convert_state_dict({"state_dict": {"other.weight": torch.zeros(1)}})


def test_frame_uses_first_last_not_average():
    images = torch.rand(2, 3, 3, 28, 28)
    poses = torch.eye(4).repeat(2, 3, 1, 1)
    poses[:, :, 0, 3] = torch.tensor([[10, 20, 12], [-2, 5, 2]])
    processor = C3GProcessor(image_shape=[28, 28])
    prepared = processor.preprocess(Views(images, c2w=poses), torch.device("cpu"))
    assert_f32_close(prepared.frame.anchor_c2w, poses[:, 0])
    assert_f32_close(prepared.frame.scale, torch.tensor([0.5, 0.25]))
    no_poses = processor.preprocess(Views(images), torch.device("cpu"))
    assert torch.equal(prepared.model_input, no_poses.model_input)
    assert processor.camera_input == "none"
    assert processor.render_planes(prepared.frame) == (0.1, 100.0)
    assert processor.background == (0.0, 0.0, 0.0)
    assert processor.render_settings == {
        "radius_clip": 0.0,
        "rasterize_mode": "classic",
        "clamp": False,
    }
    assert "eps2d" not in processor.render_settings


@pytest.mark.parametrize("baseline", [0.0, 1e-4, 1e11])
def test_invalid_baseline(baseline):
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[-1, 0, 3] = baseline
    with pytest.raises(ValueError, match="baseline"):
        C3GProcessor().preprocess(
            Views(torch.rand(2, 3, 28, 28), c2w=poses), torch.device("cpu")
        )


def test_single_view_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        C3GProcessor().preprocess(Views(torch.rand(1, 3, 28, 28)), torch.device("cpu"))


def test_pipeline_frames_and_roundtrip(tmp_path):
    model = C3G(**TINY)
    processor = C3GProcessor(image_shape=[28, 28], crop_mode="none")
    pipe = GSPipeline(model, processor)
    assert pipe.model_type == "c3g"
    assert get_model_spec("c3g").display_name == "C3G"
    images = torch.rand(1, 2, 3, 28, 28)
    output = pipe(Views(images))
    assert output.frame is None and output.cameras is None
    with pytest.raises(ValueError, match="no world frame"):
        output.to_world()
    poses = torch.eye(4).repeat(1, 2, 1, 1)
    poses[:, 0, 0, 3], poses[:, 1, 0, 3] = 10, 12
    posed = pipe(Views(images, c2w=poses))
    assert posed.cameras is None
    assert_f32_close(posed.gaussians.means, output.gaussians.means)
    world = posed.to_world()
    assert_f32_close(
        world.gaussians.means, posed.gaussians.means * 2 + torch.tensor([10, 0, 0])
    )
    assert_f32_close(world.gaussians.scales, posed.gaussians.scales * 2)
    pipe.save_pretrained(tmp_path)
    loaded = GSPipeline.from_pretrained(tmp_path)
    assert loaded.processor.to_dict() == processor.to_dict()
    assert_f32_close(loaded(Views(images)).gaussians.means, output.gaussians.means)


@pytest.mark.parametrize("shape", [[0, 224], [225, 224], [224]])
def test_bad_config(shape):
    with pytest.raises(ValueError, match="image_shape"):
        C3GProcessor(image_shape=shape)


def test_render_frame_planes_and_overrides(monkeypatch):
    calls = []

    def fake_render(gaussians, cameras, **kwargs):
        calls.append((cameras, kwargs))
        return {}

    monkeypatch.setattr(ffgs.pipeline.render_mod, "render", fake_render)
    pipe = GSPipeline(C3G(**TINY), C3GProcessor(image_shape=[28, 28]))
    images = torch.rand(1, 2, 3, 28, 28)
    poses = torch.eye(4).repeat(1, 2, 1, 1)
    poses[:, 0, 0, 3], poses[:, 1, 0, 3] = 10, 12
    cameras = Cameras(poses, torch.eye(3).repeat(1, 2, 1, 1), (28, 28), True)
    output = pipe(Views(images, c2w=poses))
    pipe.render(output, cameras)
    received, settings = calls[-1]
    assert_f32_close(received.c2w[:, :, 0, 3], torch.tensor([[0.0, 1.0]]))
    assert float(settings["near"]) == pytest.approx(0.05)
    assert float(settings["far"]) == pytest.approx(50)
    assert settings["background"] == (0, 0, 0)
    assert settings["rasterize_mode"] == "classic" and settings["clamp"] is False
    unposed = pipe(Views(images))
    pipe.render(unposed, cameras, clamp=True, background=None)
    assert calls[-1][1]["near"] == 0.1
    assert calls[-1][1]["clamp"] is True
    assert calls[-1][1]["background"] is None


@pytest.mark.gpu
def test_render_at_the_input_views():
    pipe = GSPipeline(C3G(**TINY), C3GProcessor(image_shape=[28, 28])).to("cuda")
    images = torch.rand(1, 2, 3, 28, 28, generator=torch.Generator().manual_seed(0))
    poses = torch.eye(4).repeat(1, 2, 1, 1)
    poses[:, 0, 0, 3], poses[:, 1, 0, 3] = 10, 12
    k = torch.tensor([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]]).repeat(1, 2, 1, 1)
    cameras = Cameras(poses, k, (28, 28), True)
    output = pipe(Views(images, k, poses, normalized_intrinsics=True))
    rendered = pipe.render(output, cameras)["images"]
    assert rendered.shape == (1, 2, 3, 28, 28)
    assert torch.isfinite(rendered).all()
    in_world = pipe.render(output.to_world(), cameras)["images"]
    torch.testing.assert_close(in_world, rendered, atol=1e-3, rtol=0)


@pytest.mark.network
def test_released_weights_cpu_smoke():
    """Opt-in full checkpoint load and 224-square, two-view CPU inference."""
    pipe = GSPipeline.from_pretrained("c3g/default", device="cpu")
    images = torch.rand(1, 2, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    output = pipe(Views(images))
    assert output.gaussians.means.shape == (1, 2048, 3)
    assert output.frame is None and output.cameras is None
    for name in ("means", "scales", "quats", "opacities", "colors"):
        assert torch.isfinite(getattr(output.gaussians, name)).all(), name
