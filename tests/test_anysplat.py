"""AnySplat against the official code and checkpoint.

`fixtures/anysplat_reference.safetensors` holds outputs of the official code on
tiny inputs (`fixtures/make_anysplat_reference.py`): `process_image`, the encoder
of a tiny VGGT / AnySplat with seeded weights, and the target cameras of
`eval_nvs.py`, with the float32 tokens of each aggregator call. The aggregator
runs in bfloat16, whose CPU kernels differ between machines (oneDNN with AVX-512,
others without): it is compared to bfloat16 precision, and the float32 rest of the
model, run on upstream's tokens, to float32 precision. `fixtures/anysplat_checkpoint.json` holds the tensor names, dtypes
and shapes of the released checkpoint (read from its safetensors header).
"""

from __future__ import annotations

import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from _empty import meta_parameters
from _numeric import assert_f32_close
from fixtures.make_anysplat_reference import checksum, fill_weights
from safetensors import safe_open

import ffgs.pipeline
from ffgs import Cameras, GSPipeline, Views, get_model_spec, zoo
from ffgs.geometry import quat_to_matrix
from ffgs.models.anysplat import (
    AnySplat,
    AnySplatProcessor,
    convert_state_dict,
    predict_target_cameras,
    split_llffhold,
    target_cameras,
)
from ffgs.models.anysplat.processing import quantize, upstream_scaled_shape
from ffgs.models.anysplat.vggt.layers.attention import MemEffAttention
from ffgs.models.anysplat.vggt.layers.block import Block
from ffgs.models.anysplat.vggt.layers.vision_transformer import DinoVisionTransformer

FIXTURES = Path(__file__).parent / "fixtures"
CHECKPOINT = json.loads((FIXTURES / "anysplat_checkpoint.json").read_text())
ENTRIES = zoo.list_models("anysplat")
_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32}


class _UpstreamTokens(torch.nn.Module):
    """Stands in for the aggregator: upstream's tokens of each call, by view count."""

    def __init__(self, calls: dict[int, tuple[list[torch.Tensor], int]]) -> None:
        super().__init__()
        self.calls = calls

    def forward(self, images, intermediate_layer_idx=None):
        tokens, patch_start_idx = self.calls[images.shape[1]]
        return list(tokens), patch_start_idx


class Upstream:
    def __init__(self) -> None:
        path = FIXTURES / "anysplat_reference.safetensors"
        with safe_open(str(path), framework="pt") as handle:
            self.meta = handle.metadata()
            self.tensors = {k: handle.get_tensor(k) for k in handle.keys()}
        self.tiny = json.loads(self.meta["tiny"])
        self.dino = json.loads(self.meta["dino"])
        self.cases = json.loads(self.meta["cases"])
        self.sources = json.loads(self.meta["sources"])
        self.stride = json.loads(self.meta["stride"])

    def case(self, name: str) -> dict:
        prefix = f"{name}."
        out = {
            k[len(prefix) :]: v
            for k, v in self.tensors.items()
            if k[: len(prefix)] == prefix
        }
        out.update(self.cases[name])
        return out

    def tokens(self, key: str) -> list[torch.Tensor]:
        return [self.tensors[f"{key}.tokens.{i}"] for i in range(4)]

    def model(self, name: str, upstream_tokens: bool = False) -> AnySplat:
        """The tiny model of case `name`, with the generator's weights; with
        `upstream_tokens`, its aggregator returns upstream's tokens instead."""
        model = AnySplat(voxel_size=self.cases[name]["voxel_size"], **self.tiny)
        dino = DinoVisionTransformer(
            block_fn=partial(Block, attn_class=MemEffAttention), **self.dino
        )
        model.encoder.aggregator.patch_embed = dino.to(torch.bfloat16)
        fill_weights(model.encoder)
        want = json.loads(self.meta[f"{name}.checksum"])
        assert checksum(model.encoder) == pytest.approx(want, rel=1e-12)
        if upstream_tokens:
            start = json.loads(self.meta[f"{name}.patch_start_idx"])
            calls = {self.cases[name]["views"]: (self.tokens(name), start)}
            if f"{name}.nvs.tokens.0" in self.tensors:
                nvs = self.tokens(f"{name}.nvs")
                calls[nvs[0].shape[1]] = (nvs, start)
            model.encoder.aggregator = _UpstreamTokens(calls)
        return model.eval()

    def processor(self, name: str, **overrides) -> AnySplatProcessor:
        shape = list(self.cases[name]["shape"])
        return AnySplatProcessor(image_shape=shape, crop_mode="none", **overrides)


@pytest.fixture(scope="module")
def upstream() -> Upstream:
    return Upstream()


def _model_input(images: torch.Tensor) -> torch.Tensor:
    # What the processor makes of 8-bit images already at the model's shape.
    return (quantize(images) * 2.0 - 1.0 + 1) * 0.5


# --- the official code ------------------------------------------------------------


@pytest.mark.parametrize("index", [0, 1, 2])
def test_preprocess_matches_process_image(upstream, index) -> None:
    # 8-bit sources, resized (PIL bicubic, int-truncated side) and cropped to 448.
    source = upstream.tensors[f"process.{index}.source"]
    prepared = AnySplatProcessor().preprocess(Views(source[None, None]), "cpu")
    assert prepared.frame is None
    images = prepared.model_input[0, 0]
    digest = hashlib.sha256(images.contiguous().numpy().tobytes()).digest()
    assert digest == bytes(upstream.tensors[f"process.{index}.sha256"].tolist())
    stride = upstream.stride
    want = upstream.tensors[f"process.{index}.subsample"]
    assert torch.equal(images[:, ::stride, ::stride], want)


def test_scaled_shape_truncates_as_upstream(upstream) -> None:
    shapes = [upstream_scaled_shape(s, (448, 448)) for s in upstream.sources]
    assert shapes == [(448, 664), (700, 448), (448, 448)]  # round: 665, 701


@pytest.mark.parametrize("name", ["single", "batch"])
def test_aggregator_matches_upstream(upstream, name) -> None:
    # To bfloat16 precision: running it in float32 instead moves the tokens by
    # under 1e-2 of their norm; a wrong layer or token order moves them by ~1.
    case = upstream.case(name)
    encoder = upstream.model(name).encoder
    with torch.no_grad():
        tokens, patch_start_idx = encoder.aggregator(
            case["images"].to(torch.bfloat16),
            intermediate_layer_idx=encoder.cfg.intermediate_layer_idx,
        )
    assert patch_start_idx == json.loads(upstream.meta[f"{name}.patch_start_idx"])
    for got, want in zip(tokens, upstream.tokens(name), strict=True):
        assert got.shape == want.shape
        assert (got.float() - want).norm() / want.norm() < 3e-2


@pytest.mark.parametrize("name", ["single", "batch"])
def test_forward_matches_upstream(upstream, name) -> None:
    # From upstream's tokens on: the merged voxels, and in "batch" the padded samples.
    case = upstream.case(name)
    with torch.no_grad():
        out = upstream.model(name, upstream_tokens=True)(case["images"])
    g = out.gaussians
    got = {
        "means": g.means,
        "harmonics": g.harmonics,
        "opacities": g.opacities,
        "scales": g.scales,
        "rotations": g.rotations,
        "c2w": out.pred_context_pose["extrinsic"],
        "intrinsics": out.pred_context_pose["intrinsic"],
        "depth": out.depth_dict["depth"],
    }
    for key, value in got.items():
        assert_f32_close(value, case[key], msg=key, padding=-1e4)


def test_target_cameras_match_eval_nvs(upstream) -> None:
    case = upstream.case("single")
    model = upstream.model("single", upstream_tokens=True)
    cameras = predict_target_cameras(
        model, case["images"], case["nvs.targets"], case["c2w"]
    )
    assert_f32_close(cameras.c2w, case["nvs.c2w"])
    assert_f32_close(cameras.intrinsics, case["nvs.intrinsics"])
    assert cameras.normalized_intrinsics
    assert cameras.image_shape == tuple(case["shape"])


# --- the pipeline -----------------------------------------------------------------


def test_pipeline_returns_upstream_gaussians_in_the_model_frame(upstream) -> None:
    case = upstream.case("single")
    model = upstream.model("single")
    pipe = GSPipeline(model, upstream.processor("single"))
    assert pipe.model_type == "anysplat"
    out = pipe(Views(case["images"]))
    with torch.no_grad():
        raw = model(_model_input(case["images"]))
    g = out.gaussians
    assert torch.equal(g.means, raw.gaussians.means)
    assert torch.equal(g.scales, raw.gaussians.scales)
    assert torch.equal(g.opacities, raw.gaussians.opacities)
    assert torch.equal(g.quats, raw.gaussians.rotations[..., [3, 0, 1, 2]])
    assert torch.equal(g.colors, raw.gaussians.harmonics.transpose(-1, -2))
    assert g.sh_degree == 4 and g.colors.shape[-2:] == (25, 3)
    # Same rotation either way round (xyzw upstream, wxyz here).
    x, y, z, w = raw.gaussians.rotations.unbind(-1)
    torch.testing.assert_close(
        quat_to_matrix(torch.stack([w, x, y, z], -1)), quat_to_matrix(g.quats)
    )
    assert out.space == "model" and out.frame is None
    assert torch.equal(out.cameras.c2w, raw.pred_context_pose["extrinsic"])
    assert torch.equal(out.cameras.intrinsics, raw.pred_context_pose["intrinsic"])
    assert out.cameras.image_shape == tuple(case["shape"])
    with pytest.raises(ValueError, match="no world frame"):
        out.to_world()


def test_batch_padding_is_inert(upstream, tmp_path) -> None:
    # Upstream pads the smaller sample to the batch's Gaussian count with
    # placeholders (means -1e4, scales 0); ffgs keeps the real Gaussians and makes
    # the placeholders transparent copies of the sample's first.
    case = upstream.case("batch")
    with torch.no_grad():
        raw = upstream.model("batch", upstream_tokens=True)(case["images"])
    counts = raw.infos["num_gaussians"]
    padded = (case["means"] == -1e4).all(-1)
    assert counts.tolist() == (~padded).sum(1).tolist()
    assert padded.any(), "the fixture case should pad"
    g = AnySplatProcessor().postprocess(raw)
    for b, n in enumerate(counts.tolist()):
        assert_f32_close(g.means[b, :n], case["means"][b, :n])
        assert_f32_close(g.scales[b, :n], case["scales"][b, :n])
        assert_f32_close(g.opacities[b, :n], case["opacities"][b, :n])
        want = case["harmonics"][b, :n].transpose(-1, -2)
        assert_f32_close(g.colors[b, :n], want)
        rest = slice(n, None)
        assert (g.opacities[b, rest] == 0).all() and (g.colors[b, rest] == 0).all()
        assert torch.equal(g.means[b, rest], g.means[b, :1].expand_as(g.means[b, rest]))
        assert (g.scales[b, rest] > 0).all()
        path = g.save_ply(tmp_path / f"{b}.ply", index=b)
        data = path.read_bytes()
        assert f"element vertex {n}\n".encode() in data  # padding left out
        body = data[data.index(b"end_header\n") + len(b"end_header\n") :]
        values = np.frombuffer(body, dtype="<f4")
        assert np.isfinite(values).all()


def test_a_batch_of_one_has_no_padding(upstream) -> None:
    case = upstream.case("single")
    with torch.no_grad():
        raw = upstream.model("single", upstream_tokens=True)(case["images"])
    assert raw.infos["num_gaussians"].tolist() == [case["means"].shape[1]]
    g = AnySplatProcessor().postprocess(raw)
    assert_f32_close(g.means, case["means"])
    assert_f32_close(g.opacities, case["opacities"])


def test_poses_fit_the_frame(upstream) -> None:
    case = upstream.case("single")
    pipe = GSPipeline(upstream.model("single"), upstream.processor("single"))
    predicted = pipe(Views(case["images"])).cameras.c2w
    # The given poses: the predicted ones through a known similarity.
    angle = torch.tensor(0.7)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    given = predicted.clone().double()
    given[..., :3, :3] = rotation.double() @ given[..., :3, :3]
    given[..., :3, 3] = 2.5 * given[..., :3, 3] @ rotation.double().T + 1.0
    out = pipe(Views(case["images"], c2w=given.float()))
    assert out.frame is not None
    assert out.frame.residuals.rotation_deg.max() < 1e-2
    world = out.to_world()
    torch.testing.assert_close(world.cameras.c2w, given.float(), atol=1e-4, rtol=0)


def test_render_uses_the_processor_settings(upstream, monkeypatch) -> None:
    seen = []

    def fake_render(gaussians, cameras, **kwargs):
        seen.append(kwargs)
        return {}

    monkeypatch.setattr(ffgs.pipeline.render_mod, "render", fake_render)
    case = upstream.case("single")
    pipe = GSPipeline(upstream.model("single"), upstream.processor("single"))
    out = pipe(Views(case["images"]))
    pipe.render(out, out.cameras)
    assert seen[-1] == {
        "near": 1e-10,
        "far": 1e10,
        "background": (1.0, 1.0, 1.0),
        "radius_clip": 0.1,
        "rasterize_mode": "classic",
        "clamp": True,
    }
    # The caller's arguments win.
    pipe.render(out, out.cameras, radius_clip=0.0, background=None)
    assert seen[-1]["radius_clip"] == 0.0 and seen[-1]["background"] is None
    # With poses (here: the predicted ones, 4 times as far apart), the planes go
    # to world units and back: the rasteriser still sees upstream's model units.
    c2w = out.cameras.c2w.clone()
    c2w[..., :3, 3] *= 4.0
    posed = pipe(Views(case["images"], c2w=c2w))
    torch.testing.assert_close(posed.frame.scale, torch.tensor([0.25]))
    pipe.render(posed, Cameras(c2w, out.cameras.intrinsics, (28, 42), True))
    assert float(seen[-1]["near"]) == pytest.approx(1e-10)
    assert float(seen[-1]["far"]) == pytest.approx(1e10)


def test_target_cameras_through_the_pipeline(upstream) -> None:
    case = upstream.case("single")
    model = upstream.model("single")
    pipe = GSPipeline(model, upstream.processor("single"))
    context, targets = Views(case["images"]), Views(case["nvs.targets"])
    out = pipe(context)
    cameras = target_cameras(pipe, out, context, targets)
    want = predict_target_cameras(
        model,
        _model_input(case["images"]),
        _model_input(case["nvs.targets"]),
        out.cameras.c2w,
    )
    assert torch.equal(cameras.c2w, want.c2w)
    assert torch.equal(cameras.intrinsics, want.intrinsics)
    posed = pipe(Views(case["images"], c2w=out.cameras.c2w))
    with pytest.raises(ValueError, match="without c2w"):
        target_cameras(pipe, posed, context, targets)


def test_split_llffhold() -> None:
    assert split_llffhold(10) == ([1, 2, 3, 4, 5, 6, 7, 9], [0, 8])
    assert split_llffhold(5, llffhold=2) == ([1, 3], [0, 2, 4])


def test_save_and_reload_round_trips(upstream, tmp_path) -> None:
    # The architecture arguments travel in config.json (here: the tiny model with
    # its convolutional patch embedding).
    model = AnySplat(voxel_size=0.05, **upstream.tiny).eval()
    fill_weights(model.encoder)
    pipe = GSPipeline(model, upstream.processor("single"))
    pipe.save_pretrained(tmp_path)
    loaded = GSPipeline.from_pretrained(tmp_path)
    assert isinstance(loaded.model, AnySplat)
    assert loaded.processor.config == pipe.processor.config
    images = upstream.case("single")["images"]
    a, b = pipe(Views(images)), loaded(Views(images))
    assert torch.equal(a.gaussians.means, b.gaussians.means)
    assert torch.equal(a.gaussians.colors, b.gaussians.colors)


# --- the processor ----------------------------------------------------------------


def _posed_views(shape: tuple[int, int]) -> Views:
    g = torch.Generator().manual_seed(0)
    h, w = shape
    k = torch.tensor([[0.9 * w, 0, 0.47 * w], [0, 1.1 * h, 0.52 * h], [0, 0, 1]])
    return Views(
        torch.rand(1, 2, 3, h, w, generator=g),
        k.expand(1, 2, 3, 3),
        torch.eye(4).expand(1, 2, 4, 4),
    )


@pytest.mark.parametrize("crop_mode", ["crop", "pad"])
@pytest.mark.parametrize("shape", [(60, 89), (97, 62)])
def test_fit_cameras_matches_fit_views(crop_mode, shape) -> None:
    processor = AnySplatProcessor(image_shape=[56, 42], crop_mode=crop_mode)
    views = _posed_views(shape)
    fitted = processor.fit_views(views)
    assert fitted.image_shape == (56, 42)
    cameras = processor.fit_cameras(views.cameras)
    assert cameras.image_shape == (56, 42)
    torch.testing.assert_close(cameras.intrinsics, fitted.intrinsics)


def test_crop_keeps_the_normalised_principal_point() -> None:
    views = _posed_views((60, 89))
    fitted = AnySplatProcessor().fit_views(views)
    k, src = fitted.intrinsics[0, 0], views.normalized_k[0, 0]
    assert torch.equal(k[:2, 2], src[:2, 2])
    # fx scales with the resized width over the crop: 664 / 448.
    torch.testing.assert_close(k[0, 0], src[0, 0] * 664 / 448)
    torch.testing.assert_close(k[1, 1], src[1, 1])


def test_inputs_are_quantised_to_8_bit() -> None:
    images = torch.rand(1, 1, 3, 28, 42)
    processor = AnySplatProcessor(image_shape=[28, 42], crop_mode="none")
    fitted = processor.fit_views(Views(images)).images
    assert torch.equal(fitted, (images * 255).round() / 255)


def test_other_resize_modes_and_a_custom_resize() -> None:
    views = _posed_views((60, 89))
    for mode in ("bilinear", "lanczos"):
        fitted = AnySplatProcessor(resize_mode=mode).fit_views(views)
        assert fitted.image_shape == (448, 448)
    calls = []

    def resize(images, shape):
        calls.append(shape)
        return torch.zeros(*images.shape[:-2], *shape)

    processor = AnySplatProcessor(resize=resize)
    assert processor.fit_views(views).images.abs().max() == 0
    assert calls == [(448, 664)]


def test_processor_config_is_validated() -> None:
    with pytest.raises(ValueError, match="multiples of 14"):
        AnySplatProcessor(image_shape=[448, 450])
    with pytest.raises(ValueError, match="resize_mode"):
        AnySplatProcessor(resize_mode="nearest")
    with pytest.raises(ValueError, match="crop_mode"):
        AnySplatProcessor(crop_mode="zoom")
    with pytest.raises(ValueError, match="rasterize_mode"):
        AnySplatProcessor(rasterize_mode="fancy")


# --- the checkpoint and the zoo -----------------------------------------------------


def test_every_anysplat_entry_points_at_the_checkpoint() -> None:
    assert ENTRIES == ["anysplat/default"]
    for name in ENTRIES:
        weights = zoo.get_entry(name).weights
        assert (weights.repo, weights.file, weights.revision) == (
            CHECKPOINT["repo"],
            CHECKPOINT["file"],
            CHECKPOINT["revision"],
        )
        assert "CC BY-NC 4.0" in zoo.get_entry(name).license


@pytest.mark.parametrize("name", ENTRIES)
def test_entry_model_has_the_checkpoint_layout(name) -> None:
    # Every name, dtype and shape of the released checkpoint, and nothing else.
    entry = zoo.get_entry(name)
    spec = get_model_spec(entry.model_type)
    with meta_parameters():
        model = spec.model_cls(**entry.model)
    state = model.state_dict()
    layout = {k: [str(v.dtype), list(v.shape)] for k, v in state.items()}
    want = {
        k: [str(_DTYPES[dtype]), shape]
        for k, (dtype, shape) in CHECKPOINT["tensors"].items()
    }
    assert layout == want
    checkpoint = {
        k: torch.empty(shape, dtype=_DTYPES[dtype], device="meta")
        for k, (dtype, shape) in CHECKPOINT["tensors"].items()
    }
    model.load_state_dict(convert_state_dict(checkpoint), strict=True, assign=True)


@pytest.mark.network
def test_checkpoint_layout_is_the_published_one() -> None:
    # Reads only the safetensors header of the pinned file.
    from test_tokengs import _safetensors_header

    header = _safetensors_header(
        CHECKPOINT["repo"], CHECKPOINT["file"], CHECKPOINT["revision"]
    )
    got = {k: [v["dtype"], v["shape"]] for k, v in header.items()}
    assert got == CHECKPOINT["tensors"]


@pytest.mark.network
def test_released_model_runs_on_cpu() -> None:
    # The real weights (2.9 GB download), loaded without a second copy in memory,
    # on two small views.
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    entry = zoo.get_entry("anysplat/default")
    path = hf_hub_download(
        entry.weights.repo, entry.weights.file, revision=entry.weights.revision
    )
    with meta_parameters():
        model = AnySplat(**entry.model)
    model.load_state_dict(convert_state_dict(load_file(path)), strict=True, assign=True)
    processor = AnySplatProcessor.from_dict(
        {**entry.processor, "image_shape": [224, 224]}
    )
    pipe = GSPipeline(model.eval(), processor)
    g = torch.Generator().manual_seed(0)
    images = torch.rand(1, 1, 3, 32, 32, generator=g)
    images = torch.nn.functional.interpolate(
        images[0], size=(300, 400), mode="bilinear"
    ).expand(2, 3, 300, 400)[None]
    images = images + 0.02 * torch.randn(images.shape, generator=g)
    out = pipe(Views(images.clamp(0, 1)))
    gaussians = out.gaussians
    assert gaussians.sh_degree == 4
    assert 0 < gaussians.num_gaussians <= 2 * 224 * 224
    for value in (gaussians.means, gaussians.scales, gaussians.colors):
        assert torch.isfinite(value).all()
    assert ((gaussians.opacities >= 0) & (gaussians.opacities <= 1)).all()
    # The camera head is trained to predict the first camera as the identity
    # (VGGT's frame); it does so only approximately (here within 0.02).
    torch.testing.assert_close(out.cameras.c2w[0, 0], torch.eye(4), atol=0.05, rtol=0)
