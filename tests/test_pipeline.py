import json

import pytest
import torch
from _dummy import MODEL_TYPE, SPEC, DummyModel, DummyProcessor
from test_geometry import eval_sh

from ffgs import (
    Cameras,
    GSOutput,
    GSPipeline,
    ModelSpec,
    Views,
    get_model_spec,
    register_model,
)
from ffgs.geometry import quat_to_matrix
from ffgs.registry import spec_for_model


def _model() -> DummyModel:
    torch.manual_seed(0)
    return DummyModel()


def _views(num: int = 3, hw: tuple[int, int] = (40, 60)) -> Views:
    g = torch.Generator().manual_seed(0)
    c2w = torch.eye(4).repeat(num, 1, 1)
    c2w[:, :3, 3] = torch.randn(num, 3, generator=g)
    k = torch.eye(3).repeat(num, 1, 1)
    k[:, 0, 0], k[:, 1, 1], k[:, :2, 2] = 0.8, 1.4, 0.5
    images = torch.randint(0, 256, (num, 3, *hw), dtype=torch.uint8, generator=g)
    return Views(images, k, c2w, normalized_intrinsics=True)


def test_pipeline_round_trip_and_forward(tmp_path) -> None:
    processor = DummyProcessor(image_shape=[16, 32], scene_scale=0.2)
    pipe = GSPipeline(_model(), processor)
    assert pipe.model_type == MODEL_TYPE
    pipe.save_pretrained(tmp_path)

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["model_type"] == MODEL_TYPE
    assert config["num_gaussians"] == 5

    loaded = GSPipeline.from_pretrained(tmp_path)
    assert loaded.processor.config == processor.config
    for key, value in pipe.model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[key], value), key
    # The model class alone still loads the directory.
    assert DummyModel.from_pretrained(tmp_path).num_gaussians == 5

    views = _views()
    out_a, out_b = pipe(views), loaded(views)
    assert out_a.space == "model"
    assert torch.equal(out_a.gaussians.means, out_b.gaussians.means)
    # The pipeline's output is the model's own on the prepared input.
    prepared = processor.preprocess(views, torch.device("cpu"))
    with torch.no_grad():
        raw = processor.postprocess(pipe.model(prepared.model_input))
    assert torch.equal(out_a.gaussians.means, raw.means)
    assert torch.equal(out_a.gaussians.colors, raw.colors)
    assert processor.render_planes(out_a.frame) == (0.025 / 0.2, 125.0 / 0.2)


def test_to_world_applies_the_frame() -> None:
    pipe = GSPipeline(_model(), DummyProcessor(scene_scale=0.2))
    out = pipe(_views())
    world = out.to_world()
    assert (world.space, world.frame) == ("world", out.frame)
    want = out.frame.model_to_world().apply(out.gaussians)
    assert torch.equal(world.gaussians.means, want.means)
    assert torch.equal(world.gaussians.colors, want.colors)
    assert world.gaussians.means.dtype == torch.float32
    assert world.to_world() is world
    with pytest.raises(ValueError, match="space"):
        GSOutput(out.gaussians, out.frame, space="camera")


def _fake_render(gaussians, cameras, *, near, far, background=None, **kwargs):
    """What a rasteriser sees of each Gaussian in each view: pixel, depth over
    near / far, camera-space covariance over depth^2, view-dependent colour."""
    c2w = cameras.c2w.double()
    viewmats = torch.inverse(c2w)
    means = gaussians.means.double()
    cam = torch.einsum("bvij,bnj->bvni", viewmats[..., :3, :3], means)
    cam = cam + viewmats[:, :, None, :3, 3]
    depth = cam[..., 2:]
    pixels = torch.einsum("bvij,bvnj->bvni", cameras.pixel_k.double(), cam / depth)
    r = quat_to_matrix(gaussians.quats.double())
    cov = r @ torch.diag_embed(gaussians.scales.double() ** 2) @ r.transpose(-1, -2)
    w2c = viewmats[..., None, :3, :3]
    cov = w2c @ cov[:, None] @ w2c.transpose(-1, -2) / depth[..., None] ** 2
    dirs = torch.nn.functional.normalize(
        means[:, None] - c2w[:, :, None, :3, 3], dim=-1
    )
    colors = eval_sh(gaussians.colors.double()[:, None], dirs)
    if isinstance(near, torch.Tensor):  # one per sample
        near, far = (
            near.double()[:, None, None, None],
            far.double()[:, None, None, None],
        )
    return {
        "pixels": pixels[..., :2],
        "near": depth / near,
        "far": depth / far,
        "cov": cov,
        "colors": colors,
    }


def test_render_is_the_same_in_either_space(monkeypatch) -> None:
    torch.manual_seed(1)
    model = DummyModel(num_gaussians=20, sh_degree=3)
    pipe = GSPipeline(model, DummyProcessor(scene_scale=0.2))
    out = pipe(_views())
    out.gaussians.means[..., 2] += 2.0  # in front of the cameras
    monkeypatch.setattr("ffgs.pipeline.render_mod.render", _fake_render)
    g = torch.Generator().manual_seed(2)
    c2w = torch.eye(4).repeat(1, 2, 1, 1)
    c2w[..., :3, 3] = torch.randn(1, 2, 3, generator=g) * 0.1
    cameras = Cameras(c2w, torch.eye(3).repeat(1, 2, 1, 1), (8, 8), True)
    in_model = pipe.render(out, cameras)
    in_world = pipe.render(out.to_world(), cameras)
    for key, value in in_model.items():
        torch.testing.assert_close(value, in_world[key], atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
def test_gsplat_render_is_the_same_in_either_space() -> None:
    torch.manual_seed(1)
    pipe = GSPipeline(DummyModel(num_gaussians=200, sh_degree=3), DummyProcessor())
    pipe.to("cuda")
    views = _views()
    out = pipe(views)
    out.gaussians.means[..., 2] = out.gaussians.means[..., 2].abs() + 1.0
    out.gaussians.scales.clamp_(max=0.3)
    cameras = Cameras(views.c2w, views.intrinsics, views.image_shape, True)
    in_model = pipe.render(out, cameras)["images"]
    in_world = pipe.render(out.to_world(), cameras)["images"]
    assert (in_model - 0.5).abs().max() > 0.01  # something was drawn
    torch.testing.assert_close(in_model, in_world, atol=1e-3, rtol=0)


def test_processor_overrides_take_a_resize_callable(tmp_path) -> None:
    GSPipeline(_model(), DummyProcessor()).save_pretrained(tmp_path)

    def resize(images, shape):
        return images.new_zeros((*images.shape[:-2], *shape))

    loaded = GSPipeline.from_pretrained(
        tmp_path, processor_overrides={"crop_mode": "pad", "resize": resize}
    )
    assert loaded.processor.config.crop_mode == "pad"
    assert loaded.processor.resize is resize


def test_call_options_reach_the_processor() -> None:
    pipe = GSPipeline(_model(), DummyProcessor(scene_scale=0.2))
    assert pipe(_views(), scene_scale=0.25).frame.scale == 0.25


def test_processor_overrides_and_unknown_keys(tmp_path) -> None:
    pipe = GSPipeline(_model(), DummyProcessor(image_shape=[16, 32]))
    pipe.save_pretrained(tmp_path)
    loaded = GSPipeline.from_pretrained(
        tmp_path, processor_overrides={"scene_scale": 0.25}
    )
    assert loaded.processor.config.scene_scale == 0.25
    data = json.loads((tmp_path / "processor_config.json").read_text())
    data["bogus"] = 1
    (tmp_path / "processor_config.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="bogus"):
        GSPipeline.from_pretrained(tmp_path)


def test_missing_model_type_needs_an_explicit_one(tmp_path) -> None:
    _model().save_pretrained(tmp_path)
    DummyProcessor().save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="model_type"):
        GSPipeline.from_pretrained(tmp_path)
    assert GSPipeline.from_pretrained(tmp_path, model_type=MODEL_TYPE).model_type


def test_missing_processor_config_is_reported(tmp_path) -> None:
    _model().save_pretrained(tmp_path)
    with pytest.raises(FileNotFoundError, match="processor_config.json"):
        GSPipeline.from_pretrained(tmp_path, model_type=MODEL_TYPE)


def test_registry() -> None:
    assert get_model_spec(MODEL_TYPE) is SPEC
    assert spec_for_model(_model()) is SPEC
    assert register_model(SPEC) is SPEC  # re-registering the same spec is a no-op
    with pytest.raises(ValueError, match="already registered"):
        register_model(ModelSpec(MODEL_TYPE, torch.nn.Linear, DummyProcessor))
    with pytest.raises(KeyError, match="unknown model_type"):
        get_model_spec("no-such-model")
    with pytest.raises(KeyError, match="no model_type registered"):
        spec_for_model(torch.nn.Linear(1, 1))


def test_device_of_a_model_without_parameters() -> None:
    cpu = GSPipeline(torch.nn.Identity(), DummyProcessor(), model_type="identity")
    assert cpu.device == torch.device("cpu")
    model = torch.nn.Module()
    model.register_buffer("table", torch.zeros(2, device="meta"))
    meta = GSPipeline(model, DummyProcessor(), model_type="buffers")
    assert meta.device == torch.device("meta")
