"""Generate CPU fixtures using only the official C3G source, never the ffgs port.

Run from ffgs using an existing reference environment with jaxtyping/einops:
  uv run --no-project --python /path/to/reference-env/bin/python \
    tests/fixtures/make_c3g_reference.py --upstream ~/work/3dv/C3G \
    --out tests/fixtures/c3g_reference.safetensors --checkpoint /path/to/ckpt

The official EncoderVGGT.forward, Gaussian adapter, normalization, crop shim,
VGGT aggregator and DINO patch embedding execute on CPU with tiny dimensions.
Only training imports are stubbed. A constructed tiny encoder bypasses the
hardcoded 2048-dimensional constructor, not the official forward. Visualization
DPT is replaced by a zero-output module: it cannot affect Gaussian outputs.
Released checkpoint metadata is read from the FULL file with weights_only=True,
and its complete sha256 is recomputed. No checkpoint tensors are redistributed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import types
import zlib
from pathlib import Path
from types import SimpleNamespace

import PIL
import torch
from safetensors.torch import save_file
from torch import nn

UPSTREAM = "40b2cc9"
TINY = {
    "num_gaussians": 8,
    "embed_dim": 16,
    "decoder_depth": 2,
    "decoder_heads": 2,
    "aggregator": {"img_size": 56, "depth": 4, "num_heads": 2, "patch_embed": "conv"},
    "dpt_head": {
        "features": 8,
        "out_channels": [4, 8, 8, 8],
        "intermediate_layer_idx": [0, 1, 2, 3],
    },
}
DINO = {
    "img_size": 56,
    "patch_size": 14,
    "embed_dim": 16,
    "depth": 1,
    "num_heads": 2,
    "mlp_ratio": 4,
    "num_register_tokens": 4,
    "interpolate_antialias": True,
    "interpolate_offset": 0.0,
    "block_chunks": 0,
    "init_values": 1.0,
}
CASES = {"pair": (1, 2, 28, 42), "batch": (2, 3, 42, 28)}
SOURCES = ((360, 640), (641, 359), (301, 301))
STRIDE = 4


def preprocessing_source(index: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20261016 + index)
    return torch.rand(3, *SOURCES[index], generator=generator)


def fill_weights(module: nn.Module) -> None:
    """Parameter-name seeded weights, independent of constructor RNG order."""
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            if not parameter.is_floating_point():
                continue
            generator = torch.Generator().manual_seed(zlib.crc32(name.encode()))
            values = torch.randn(parameter.shape, generator=generator)
            if parameter.ndim >= 2:
                values /= math.sqrt(parameter[0].numel())
            elif "norm" in name and name.endswith("weight"):
                values = 1 + 0.1 * values
            else:
                values *= 0.1
            parameter.copy_(values)


def install_shims(upstream: Path):
    """Load files without executing upstream training package initializers."""
    packages = [
        "src",
        "src.model",
        "src.model.encoder",
        "src.model.encoder.common",
        "src.model.encoder.backbone",
        "src.model.encoder.backbone.vggt",
        "src.model.encoder.backbone.vggt.heads",
        "src.model.encoder.heads",
        "src.model.encoder.backbone.croco",
        "src.dataset",
        "src.dataset.shims",
        "src.geometry",
        "src.misc",
    ]
    for name in packages:
        module = types.ModuleType(name)
        module.__path__ = [str(upstream.joinpath(*name.split(".")))]
        sys.modules[name] = module
    stubs = {
        "src.dataset.types": dict(
            BatchedExample=dict,
            BatchedViews=dict,
            AnyExample=dict,
            AnyViews=dict,
            DataShim=object,
        ),
        "src.geometry.projection": dict(sample_image_grid=None, get_world_rays=None),
        "src.misc.sh_rotation": dict(rotate_sh=None),
        "src.model.encoder.backbone.croco.misc": dict(
            fill_default_args=None, freeze_all_params=None
        ),
    }
    for name, attributes in stubs.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        sys.modules[name] = module
    backbone = sys.modules["src.model.encoder.backbone"]
    backbone.Backbone = nn.Module
    backbone.BackboneCfg = object
    backbone.get_backbone = None
    heads = sys.modules["src.model.encoder.heads"]
    heads.DPTHead = None
    return importlib.import_module("src.model.encoder.encoder_vggt")


class VisualizationOnly(nn.Module):
    def forward(self, tokens, images, patch_start):
        return (torch.zeros(*images.shape[:2], *images.shape[-2:], 1),)


def build_official(encoder_module):
    aggregator_module = importlib.import_module(
        "src.model.encoder.backbone.vggt.aggregator"
    )
    dino_module = importlib.import_module(
        "src.model.encoder.backbone.vggt.layers.vision_transformer"
    )
    adapter_module = importlib.import_module(
        "src.model.encoder.common.gaussian_adapter"
    )
    model = encoder_module.EncoderVGGT.__new__(encoder_module.EncoderVGGT)
    nn.Module.__init__(model)
    model.cfg = SimpleNamespace(
        freeze_backbone=False,
        feature_dim=0,
        pose_free=True,
        gaussians_per_token=1,
        num_surfaces=1,
        opacity_mapping=SimpleNamespace(initial=0.0, final=0.0, warm_up=1),
    )
    model.backbone = nn.Module()
    model.backbone.aggregator = aggregator_module.Aggregator(
        embed_dim=TINY["embed_dim"], **TINY["aggregator"]
    )
    model.backbone.aggregator.patch_embed = dino_module.DinoVisionTransformer(**DINO)

    def backbone_forward(context, return_views=False):
        tokens, start = model.backbone.aggregator(context["image"])
        return tokens, None, start

    model.backbone.forward = backbone_forward
    model.dpt_head = VisualizationOnly()
    model.gaussian_tokens = nn.Parameter(torch.empty(TINY["num_gaussians"], 32))
    model.anchor_positions = nn.Parameter(
        torch.tensor([[0, 0, 1]]).repeat(TINY["num_gaussians"], 1), requires_grad=False
    )
    model.gmae_decoder = encoder_module.Transformer(32, 2, 2, 16, 64)
    model.gmae_to_gaussians = nn.Linear(32, 14)
    model.gaussian_adapter = adapter_module.UnifiedGaussianAdapter(
        adapter_module.GaussianAdapterCfg(0.5, 15.0, 0, clamping=-1)
    )
    model.raw_gs_dim = 14
    fill_weights(model)
    return model.eval()


def checkpoint_metadata(path: Path, out: Path) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    expected = "57f2f49e55b09befbe0a745b2ece3d9bd296428f605114e0db6d89cdbfc8b935"
    if digest.hexdigest() != expected:
        raise ValueError("official checkpoint sha256 mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = checkpoint["state_dict"]
    metadata = {
        "repo": "honggyuAn/C3G",
        "revision": "875e46ea830032d497ceed5aba169945afe6f8b1",
        "file": "gaussian_decoder_multiview.ckpt",
        "sha256": digest.hexdigest(),
        "size": path.stat().st_size,
        "tensors": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in state.items()
        },
    }
    out.write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    encoder_module = install_shims(args.upstream.resolve())
    crop = importlib.import_module("src.dataset.shims.crop_shim")
    normalize = importlib.import_module("src.dataset.shims.normalize_shim")
    generator = torch.Generator().manual_seed(20261016)
    tensors = {}
    for index in range(len(SOURCES)):
        source = preprocessing_source(index)
        intrinsics = torch.tensor([[0.8, 0, 0.5], [0, 1.2, 0.5], [0, 0, 1]])
        fitted, fitted_k = crop.rescale_and_crop(source, intrinsics, (224, 224))
        tensors.update(
            {
                f"process.{index}.images": normalize.normalize_image(fitted)[
                    :, ::STRIDE, ::STRIDE
                ],
                f"process.{index}.sha256": torch.tensor(
                    list(
                        hashlib.sha256(
                            (fitted * 255).round().to(torch.uint8).numpy().tobytes()
                        ).digest()
                    ),
                    dtype=torch.uint8,
                ),
                f"process.{index}.intrinsics": fitted_k,
            }
        )
    for name, (batch, views, height, width) in CASES.items():
        model = build_official(encoder_module)
        images = torch.rand(batch, views, 3, height, width, generator=generator) * 2 - 1
        adapted = []
        original_forward = model.gaussian_adapter.forward

        def capture_adapter(*inputs, **kwargs):
            output = original_forward(*inputs, **kwargs)
            adapted.append(output)
            return output

        model.gaussian_adapter.forward = capture_adapter
        with torch.no_grad():
            output = model({"image": images})
        raw = adapted[0]
        tensors.update(
            {
                f"{name}.images": images,
                f"{name}.means": output.means,
                f"{name}.colors": output.harmonics,
                f"{name}.opacities": output.opacities,
                f"{name}.scales": raw.scales.flatten(1, 3),
                f"{name}.quats": raw.rotations.flatten(1, 3)[..., [3, 0, 1, 2]],
            }
        )
    save_file(
        {name: value.contiguous() for name, value in tensors.items()},
        str(args.out),
        metadata={
            "upstream": UPSTREAM,
            "tiny": json.dumps(TINY),
            "dino": json.dumps(DINO),
            "torch": torch.__version__,
            "pillow": PIL.__version__,
            "scope": "official tiny CPU forward; visualization-only DPT bypassed",
        },
    )
    if args.checkpoint:
        checkpoint_metadata(args.checkpoint, args.out.with_name("c3g_checkpoint.json"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
