import json

import pytest
import torch
from _dummy import MODEL_TYPE, SPEC, DummyModel, DummyProcessor

from ffgs import GSPipeline, ModelSpec, Views, get_model_spec, register_model
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
    out_a = pipe(views, return_model_frame=True)
    out_b = loaded(views, return_model_frame=True)
    assert torch.equal(out_a.model_gaussians.means, out_b.model_gaussians.means)
    assert torch.equal(out_a.gaussians.colors, out_b.gaussians.colors)
    # The pipeline's model-frame output is the model's own on the prepared input.
    prepared = processor.preprocess(views, torch.device("cpu"))
    with torch.no_grad():
        raw = processor.postprocess(pipe.model(prepared.model_input))
    assert torch.equal(out_a.model_gaussians.means, raw.means)
    world = out_a.frame.gaussians_to_world(out_a.model_gaussians)
    assert torch.equal(out_a.gaussians.means, world.means)
    assert out_a.gaussians.means.dtype == torch.float32
    assert pipe(views).model_gaussians is None
    assert processor.render_planes(out_a.frame) == (0.025 / 0.2, 125.0 / 0.2)


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
