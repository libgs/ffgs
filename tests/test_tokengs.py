"""TokenGS against the official code and checkpoints.

`fixtures/tokengs_reference.safetensors` holds outputs of the official evaluation
data path and model on tiny inputs (`fixtures/make_tokengs_reference.py`);
`fixtures/tokengs_checkpoints.json` the tensor names and shapes of the official
checkpoints (read from their safetensors headers). Resampled images and
results of matrix products are compared to float32 precision: their last bits
depend on the CPU.
"""

from __future__ import annotations

import json
import struct
import urllib.request
from pathlib import Path

import pytest
import torch
from _numeric import assert_f32_close
from safetensors import safe_open

from ffgs import Cameras, GSPipeline, Views, get_model_spec, zoo
from ffgs.models.tokengs import (
    TokenGS,
    TokenGSProcessor,
    TokenGSProcessorConfig,
    convert_state_dict,
)
from ffgs.models.tokengs.processing import mean_camera, ray_condition

FIXTURES = Path(__file__).parent / "fixtures"
CHECKPOINTS = json.loads((FIXTURES / "tokengs_checkpoints.json").read_text())
ENTRIES = zoo.list_models("tokengs")


class Upstream:
    def __init__(self) -> None:
        path = FIXTURES / "tokengs_reference.safetensors"
        with safe_open(str(path), framework="pt") as handle:
            meta = handle.metadata()
            self.tensors = {k: handle.get_tensor(k) for k in handle.keys()}
        self.archs = json.loads(meta["archs"])
        self.cases = json.loads(meta["cases"])
        self.image_shape = json.loads(meta["image_shape"])
        self.scene_scale = json.loads(meta["scene_scale"])

    def case(self, name: str) -> dict:
        prefix = f"{name}."
        out = {
            k[len(prefix) :]: v
            for k, v in self.tensors.items()
            if k[: len(prefix)] == prefix
        }
        out.update(self.cases[name])
        return out

    def weights(self, name: str) -> dict[str, torch.Tensor]:
        case = self.case(name)
        return {k[8:]: v for k, v in case.items() if k.startswith("weights.")}

    def processor(self, name: str, **overrides) -> TokenGSProcessor:
        config = dict(
            image_shape=self.image_shape,
            camera_normalization=name,
            scene_scale=self.scene_scale,
            autocast=None,
        )
        return TokenGSProcessor({**config, **overrides})

    def model(self, name: str) -> TokenGS:
        model = TokenGS(**self.archs[self.cases[name]["arch"]]).eval()
        model.load_state_dict(convert_state_dict(self.weights(name)), strict=True)
        return model


@pytest.fixture(scope="module")
def upstream() -> Upstream:
    return Upstream()


def _views(case: dict, views: slice) -> Views:
    k = case["source.intrinsics"][views]
    pixel_k = torch.zeros(*k.shape[:-1], 3, 3)
    pixel_k[..., 0, 0], pixel_k[..., 1, 1] = k[..., 0], k[..., 1]
    pixel_k[..., 0, 2], pixel_k[..., 1, 2] = k[..., 2], k[..., 3]
    pixel_k[..., 2, 2] = 1
    # As the upstream loader: uint8 / 255.0 (not ffgs' uint8 conversion).
    images = case["source.images"][views] / 255.0
    return Views(images, pixel_k, case["source.c2w"][views])


CASES = ["first_cam", "mean_cam"]


@pytest.mark.parametrize("name", CASES)
def test_preprocess_matches_upstream(upstream, name) -> None:
    case = upstream.case(name)
    processor = upstream.processor(name)
    n = case["inputs"]
    prepared = processor.preprocess(_views(case, slice(0, n)), torch.device("cpu"))

    assert_f32_close(prepared.model_input["images"][0], case["input.images"])
    # Upstream computes the rays of all views (targets included) as one batch,
    # which rounds differently in the last bit; with the same batch they match.
    torch.testing.assert_close(
        prepared.model_input["plucker"][0], case["input.plucker"], rtol=0, atol=1e-6
    )
    model_c2w = prepared.frame.c2w_to_model(case["source.c2w"][None, :n])
    assert_f32_close(model_c2w[0], case["model_c2w_input"])

    # Target views: the same fit and the inputs' frame give the upstream GT and
    # cameras.
    everything = _views(case, slice(None))
    images, k = processor.fit_pixel_views(everything)
    assert_f32_close(images[0], case["images"])
    assert torch.equal(k[0], case["intrinsics"])
    c2w = prepared.frame.c2w_to_model(everything.c2w)
    cam_view = torch.inverse(c2w[0]).transpose(1, 2)
    assert_f32_close(cam_view, case["cam_view"])
    plucker = ray_condition(k, c2w, *upstream.image_shape)
    assert_f32_close(plucker[0, :n], case["input.plucker"])


@pytest.mark.parametrize("name", CASES)
def test_forward_matches_upstream(upstream, name) -> None:
    case = upstream.case(name)
    model = upstream.model(name)
    with torch.no_grad():
        gaussians = model(
            {
                "images": case["input.images"][None],
                "plucker": case["input.plucker"][None],
            }
        )
    assert_f32_close(gaussians[0], case["gaussians"])


@pytest.mark.parametrize("name", CASES)
def test_pipeline_returns_upstream_gaussians_in_the_model_frame(upstream, name) -> None:
    case = upstream.case(name)
    pipe = GSPipeline(upstream.model(name), upstream.processor(name))
    views = _views(case, slice(0, case["inputs"]))
    out = pipe(views)

    want = case["gaussians"]
    model_g = out.gaussians
    assert out.space == "model"
    assert model_g.sh_degree is None
    # Not bit-exact: the input rays differ from upstream's in the last bit (see
    # test_preprocess_matches_upstream).
    got = torch.cat(
        [
            model_g.means[0],
            model_g.opacities[0, :, None],
            model_g.scales[0],
            model_g.quats[0],
            model_g.colors[0],
        ],
        dim=-1,
    )
    torch.testing.assert_close(got, want, rtol=0, atol=1e-5)

    # World frame: the model-frame means are where the frame puts the world ones.
    world = out.to_world().gaussians
    frame = out.frame
    anchor = frame.anchor_c2w[0].double()
    means = world.means[0].double() @ anchor[:3, :3] - anchor[:3, 3] @ anchor[:3, :3]
    torch.testing.assert_close(
        (means * frame.scale).float(), model_g.means[0], atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(world.scales, model_g.scales / frame.scale)
    torch.testing.assert_close(world.colors, model_g.colors)


def test_mean_cam_frame_is_the_mean_input_camera(upstream) -> None:
    case = upstream.case("mean_cam")
    c2w = case["source.c2w"][: case["inputs"]]
    anchor = mean_camera(c2w)
    rotation = anchor[:3, :3]
    torch.testing.assert_close(rotation @ rotation.T, torch.eye(3), atol=1e-6, rtol=0)
    assert torch.det(rotation) > 0
    torch.testing.assert_close(anchor[:3, 3], c2w[:, :3, 3].mean(0))
    frame = upstream.processor("mean_cam").frame(c2w[None], 0.15)
    assert torch.equal(frame.anchor_c2w[0], anchor)


def test_save_and_reload_round_trips(upstream, tmp_path) -> None:
    case = upstream.case("mean_cam")
    pipe = GSPipeline(upstream.model("mean_cam"), upstream.processor("mean_cam"))
    pipe.save_pretrained(tmp_path)
    assert json.loads((tmp_path / "config.json").read_text())["model_type"] == "tokengs"
    loaded = GSPipeline.from_pretrained(tmp_path)
    assert isinstance(loaded.model, TokenGS)
    assert loaded.processor.config == pipe.processor.config
    views = _views(case, slice(0, case["inputs"]))
    assert torch.equal(loaded(views).gaussians.means, pipe(views).gaussians.means)


@pytest.mark.parametrize("crop_mode", ["crop", "pad"])
def test_fit_cameras_matches_fit_views(upstream, crop_mode) -> None:
    case = upstream.case("first_cam")
    processor = upstream.processor("first_cam", crop_mode=crop_mode)
    views = _views(case, slice(None))
    fitted = processor.fit_views(views)
    cameras = processor.fit_cameras(views.cameras)
    assert cameras.image_shape == fitted.image_shape == tuple(upstream.image_shape)
    torch.testing.assert_close(cameras.normalized_k, fitted.normalized_k)
    # Pixel and normalised input intrinsics give the same fit.
    normalized = Views(views.images, views.normalized_k, views.c2w, True)
    torch.testing.assert_close(
        processor.fit_views(normalized).intrinsics, fitted.intrinsics
    )


def test_pad_mode_keeps_the_whole_image(upstream) -> None:
    case = upstream.case("first_cam")
    processor = upstream.processor("first_cam", crop_mode="pad")
    prepared = processor.preprocess(_views(case, slice(0, 2)), torch.device("cpu"))
    assert prepared.model_input["images"].shape[-2:] == tuple(upstream.image_shape)
    with pytest.raises(ValueError, match="crop_mode"):
        upstream.processor("first_cam", crop_mode="none").preprocess(
            _views(case, slice(0, 2)), torch.device("cpu")
        )


def test_other_view_counts_run_but_warn(upstream) -> None:
    case = upstream.case("first_cam")
    processor = upstream.processor("first_cam", num_input_views=3)
    processor.preprocess(_views(case, slice(0, 3)), torch.device("cpu"))
    with pytest.warns(UserWarning, match="trained for 3 input views, got 2"):
        processor.preprocess(_views(case, slice(0, 2)), torch.device("cpu"))


def test_processor_config_is_validated() -> None:
    with pytest.raises(ValueError, match="camera_normalization"):
        TokenGSProcessorConfig(camera_normalization="median_cam")
    with pytest.raises(ValueError, match="autocast"):
        TokenGSProcessorConfig(autocast="fp8")
    assert TokenGSProcessor().autocast_dtype == torch.bfloat16
    assert TokenGSProcessor(autocast=None).autocast_dtype is None


def test_upstream_training_keys_are_dropped(upstream) -> None:
    weights = upstream.weights("first_cam")
    extra = {"lpips_loss.net.slice1.0.weight": torch.zeros(1)}
    assert convert_state_dict({**weights, **extra}).keys() == weights.keys()


# --- the zoo entries against the official checkpoints -----------------------------


def test_every_tokengs_entry_points_at_a_known_checkpoint() -> None:
    assert len(ENTRIES) == 10
    for name in ENTRIES:
        entry = zoo.get_entry(name)
        assert entry.weights.repo == CHECKPOINTS["repo"]
        assert entry.weights.revision == CHECKPOINTS["revision"]
        assert entry.weights.file in CHECKPOINTS["files"]


@pytest.mark.parametrize("name", ENTRIES)
def test_entry_model_has_the_checkpoint_layout(name) -> None:
    # Every official tensor maps to a parameter of the same shape, and back:
    # strict loading succeeds (checked without the weights, from their layout).
    entry = zoo.get_entry(name)
    spec = get_model_spec(entry.model_type)
    with torch.device("meta"):
        model = spec.model_cls(**entry.model)
    layout = CHECKPOINTS["layouts"][CHECKPOINTS["files"][entry.weights.file]]
    checkpoint = {k: torch.empty(shape, device="meta") for k, shape in layout.items()}
    converted = spec.convert_state_dict(checkpoint)
    got = {k: list(v.shape) for k, v in model.state_dict().items()}
    assert got == {k: list(v.shape) for k, v in converted.items()}
    assert len(got) == len(layout)


def _safetensors_header(repo: str, file: str, revision: str) -> dict:
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo, file, revision=revision)

    def fetch(first: int, last: int) -> bytes:
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={first}-{last}"}
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()

    (size,) = struct.unpack("<Q", fetch(0, 7))
    header = json.loads(fetch(8, 7 + size))
    header.pop("__metadata__", None)
    return header


@pytest.mark.network
@pytest.mark.parametrize("file", sorted(CHECKPOINTS["files"]))
def test_checkpoint_layouts_are_the_published_ones(file) -> None:
    # Reads only the safetensors header of each pinned file (a few kB).
    header = _safetensors_header(CHECKPOINTS["repo"], file, CHECKPOINTS["revision"])
    assert {v["dtype"] for v in header.values()} == {CHECKPOINTS["dtype"]}
    layout = CHECKPOINTS["layouts"][CHECKPOINTS["files"][file]]
    assert {k: v["shape"] for k, v in header.items()} == layout


@pytest.mark.gpu
def test_render_at_the_input_views(upstream) -> None:
    case = upstream.case("first_cam")
    pipe = GSPipeline(upstream.model("first_cam"), upstream.processor("first_cam"))
    pipe.to("cuda")
    views = _views(case, slice(0, case["inputs"]))
    out = pipe(views)
    fitted = pipe.processor.fit_views(views)
    cameras = Cameras(fitted.c2w, fitted.intrinsics, fitted.image_shape, True)
    images = pipe.render(out, cameras)["images"]
    assert images.shape == (1, case["inputs"], 3, *upstream.image_shape)
    assert torch.isfinite(images).all()
    in_world = pipe.render(out.to_world(), cameras)["images"]
    torch.testing.assert_close(in_world, images, atol=1e-3, rtol=0)


def test_poses_and_intrinsics_are_required(upstream) -> None:
    case = upstream.case("first_cam")
    views = _views(case, slice(0, 1))
    processor = upstream.processor("first_cam")
    with pytest.raises(ValueError, match="TokenGS needs c2w and intrinsics"):
        processor.preprocess(Views(views.images), "cpu")
    with pytest.raises(ValueError, match="TokenGS needs c2w and intrinsics"):
        processor.preprocess(Views(views.images, views.intrinsics), "cpu")
    with pytest.raises(ValueError, match="TokenGS needs intrinsics"):
        processor.fit_pixel_views(Views(views.images, c2w=views.c2w))
