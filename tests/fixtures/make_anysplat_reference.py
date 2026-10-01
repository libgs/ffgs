"""Freeze outputs of the official AnySplat code for `tests/test_anysplat.py`.

Runs upstream's `process_image` on small 8-bit images, its `EncoderAnySplat`
(`AnySplat.inference`) on tiny inputs, and the target-camera prediction of its
`src/eval_nvs.py`, and writes them to `anysplat_reference.safetensors`, with the
float32 tokens of every aggregator call. The model is
a tiny VGGT / AnySplat (same code paths, 16-dim tokens); its weights are not
stored but drawn from a seed per parameter name (`fill_weights`, which the test
repeats), with a checksum in the metadata.

Needs a checkout of https://github.com/InternRobotics/AnySplat at UPSTREAM (as the
working directory, or `--upstream`) and its inference dependencies, including
torch_scatter; xFormers is not needed (a stub is installed, as upstream falls back
without it). Upstream is run with three shims, all outside its code:

- `VGGT.from_pretrained` builds the tiny VGGT instead of downloading VGGT-1B;
- the aggregator's tokens are made float32 (what CUDA autocast gives upstream,
  needed on CPU; the ffgs model does the same);
- the Gaussian head gets the tiny `dim_in` (upstream: the literal 2048) and the
  tiny head channels.

    python tests/fixtures/make_anysplat_reference.py --upstream /path/to/AnySplat \\
        --config /path/to/lhjiang/anysplat/config.json \\
        --out tests/fixtures/anysplat_reference.safetensors

`--config` is the `config.json` of the released model (its `encoder_cfg`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import types
import zlib
from functools import partial
from pathlib import Path

import PIL
import torch
import torchvision
from PIL import Image
from safetensors.torch import save_file

UPSTREAM = "5f5e208a7dd57d52e43ea0d553a95eab526e8775"

# The tiny model, as `ffgs.models.anysplat.AnySplat` arguments; the aggregator's
# patch embedding is replaced by DINO (a tiny DinoVisionTransformer, the class of
# the released "dinov2_vitl14_reg").
TINY = dict(
    img_size=56,
    patch_size=14,
    embed_dim=16,
    intermediate_layer_idx=[0, 1, 2, 3],
    aggregator=dict(depth=4, num_heads=2, patch_embed="conv"),
    camera_head=dict(trunk_depth=1, num_heads=2),
    depth_head=dict(
        features=8, out_channels=[4, 8, 8, 8], intermediate_layer_idx=[0, 1, 2, 3]
    ),
    gs_head=dict(out_channels=[4, 8, 8, 8], intermediate_layer_idx=[0, 1, 2, 3]),
)
DINO = dict(
    img_size=56,
    patch_size=14,
    embed_dim=16,
    depth=1,
    num_heads=2,
    mlp_ratio=4,
    num_register_tokens=4,
    interpolate_antialias=True,
    interpolate_offset=0.0,
    block_chunks=0,
    init_values=1.0,
)
# Model cases: batch, views, (h, w), voxel size (released: 0.002; coarser here to
# keep the fixture small, and so that voxels merge and the two samples pad).
CASES = {
    "single": dict(batch=1, views=2, shape=(28, 42), voxel_size=0.05),
    "batch": dict(batch=2, views=3, shape=(42, 28), voxel_size=0.5),
}
NVS_TARGETS = 2  # target views of the "single" case
# process_image sources (h, w): landscape and portrait where `int` and `round`
# of the scaled side differ (664.53 -> 664, 700.9 -> 700), and a square one.
SOURCES = [(60, 89), (97, 62), (50, 50)]
CROP = 448  # process_image's output size
STRIDE = 16  # stored subsample of the process_image outputs
# The aggregator's tokens of each call, made float32 (the second shim). The
# aggregator runs in bfloat16, whose CPU kernels differ between machines; the test
# runs the float32 heads on these tokens to compare the rest to float32 precision.
TOKENS: list[tuple[list[torch.Tensor], int]] = []


def fill_weights(module: torch.nn.Module) -> None:
    """Deterministic non-trivial weights, seeded by parameter name: matrices and
    kernels ~ N(0, 1 / fan_in), norm weights 1 + 0.1 N, the rest 0.1 N."""
    with torch.no_grad():
        for name, p in module.named_parameters():
            g = torch.Generator().manual_seed(zlib.crc32(name.encode()))
            x = torch.randn(p.shape, generator=g, dtype=torch.float32)
            if p.ndim >= 2:
                x = x / math.sqrt(p[0].numel())
            elif "norm" in name and name.endswith("weight"):
                x = 1 + 0.1 * x
            else:
                x = 0.1 * x
            p.copy_(x.to(p.dtype))


def checksum(module: torch.nn.Module) -> float:
    return float(sum(p.detach().double().abs().sum() for p in module.parameters()))


def install_shims(upstream: Path):
    """Import upstream's encoder with the three shims of the module docstring."""
    xformers = types.ModuleType("xformers")
    xformers.ops = types.ModuleType("xformers.ops")
    sys.modules["xformers"], sys.modules["xformers.ops"] = xformers, xformers.ops
    sys.path.insert(0, str(upstream))
    os.chdir(upstream)
    import src.model.encoder.anysplat as A
    from src.model.encoder.vggt.heads.camera_head import CameraHead
    from src.model.encoder.vggt.heads.dpt_head import DPTHead
    from src.model.encoder.vggt.layers.attention import MemEffAttention
    from src.model.encoder.vggt.layers.block import Block
    from src.model.encoder.vggt.layers.vision_transformer import (
        DinoVisionTransformer,
    )
    from src.model.encoder.vggt.models.aggregator import Aggregator
    from src.model.encoder.vggt.models.vggt import VGGT

    e = TINY["embed_dim"]

    def tiny_from_pretrained(name):
        model = VGGT.__new__(VGGT)
        torch.nn.Module.__init__(model)
        model.aggregator = Aggregator(
            img_size=TINY["img_size"],
            patch_size=TINY["patch_size"],
            embed_dim=e,
            **TINY["aggregator"],
        )
        model.aggregator.patch_embed = DinoVisionTransformer(
            block_fn=partial(Block, attn_class=MemEffAttention), **DINO
        )
        forward = model.aggregator.forward

        def float_tokens(*args, **kwargs):
            tokens, patch_start_idx = forward(*args, **kwargs)
            tokens = [t.float() for t in tokens]
            TOKENS.append((tokens, patch_start_idx))
            return tokens, patch_start_idx

        model.aggregator.forward = float_tokens
        model.camera_head = CameraHead(dim_in=2 * e, **TINY["camera_head"])
        model.depth_head = DPTHead(
            dim_in=2 * e,
            output_dim=2,
            activation="exp",
            conf_activation="expp1",
            **TINY["depth_head"],
        )
        return model

    VGGT.from_pretrained = staticmethod(tiny_from_pretrained)
    gs_head = A.VGGT_DPT_GS_Head

    def tiny_gs_head(dim_in, **kwargs):
        return gs_head(dim_in=2 * e, **kwargs, **TINY["gs_head"])

    A.VGGT_DPT_GS_Head = tiny_gs_head
    return A


def build_encoder(A, config: dict, voxel_size: float):
    from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg

    cfg = dict(config["encoder_cfg"])
    cfg["voxel_size"] = voxel_size
    cfg["intermediate_layer_idx"] = TINY["intermediate_layer_idx"]
    cfg["gaussian_adapter"] = GaussianAdapterCfg(**cfg["gaussian_adapter"])
    cfg["opacity_mapping"] = A.OpacityMappingCfg(**cfg["opacity_mapping"])
    encoder = A.EncoderAnySplat(A.EncoderAnySplatCfg(**cfg)).eval()
    fill_weights(encoder)
    return encoder


def eval_nvs_target_cameras(model_encoder, ctx_images, tgt_images, pred_context_pose):
    """Upstream `src/eval_nvs.py:evaluate` from the second aggregator pass to the
    rescaled target cameras: its statements, with `model.encoder` passed in, the
    `print` and the unused context-intrinsics slice dropped."""
    from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri

    b, v, _, h, w = tgt_images.shape
    num_context_view = ctx_images.shape[1]
    vggt_input_image = torch.cat((ctx_images, tgt_images), dim=1).to(torch.bfloat16)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
        aggregated_tokens_list, patch_start_idx = model_encoder.aggregator(
            vggt_input_image,
            intermediate_layer_idx=model_encoder.cfg.intermediate_layer_idx,
        )
    with torch.cuda.amp.autocast(enabled=False):
        fp32_tokens = [token.float() for token in aggregated_tokens_list]
        pred_all_pose_enc = model_encoder.camera_head(fp32_tokens)[-1]
        pred_all_extrinsic, pred_all_intrinsic = pose_encoding_to_extri_intri(
            pred_all_pose_enc, vggt_input_image.shape[-2:]
        )

    extrinsic_padding = (
        torch.tensor(
            [0, 0, 0, 1],
            device=pred_all_extrinsic.device,
            dtype=pred_all_extrinsic.dtype,
        )
        .view(1, 1, 1, 4)
        .repeat(b, vggt_input_image.shape[1], 1, 1)
    )
    pred_all_extrinsic = torch.cat(
        [pred_all_extrinsic, extrinsic_padding], dim=2
    ).inverse()

    pred_all_intrinsic[:, :, 0] = pred_all_intrinsic[:, :, 0] / w
    pred_all_intrinsic[:, :, 1] = pred_all_intrinsic[:, :, 1] / h
    pred_all_target_extrinsic = pred_all_extrinsic[:, num_context_view:]
    pred_all_context_extrinsic = pred_all_extrinsic[:, :num_context_view]
    pred_all_target_intrinsic = pred_all_intrinsic[:, num_context_view:]

    scale_factor = (
        pred_context_pose["extrinsic"][:, :, :3, 3].mean()
        / pred_all_context_extrinsic[:, :, :3, 3].mean()
    )
    pred_all_target_extrinsic[..., :3, 3] = (
        pred_all_target_extrinsic[..., :3, 3] * scale_factor
    )
    return pred_all_target_extrinsic, pred_all_target_intrinsic.float()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True, help="HF config.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out_path = args.out.resolve()
    config = json.loads(args.config.read_text())
    A = install_shims(args.upstream.resolve())
    from src.utils.image import process_image

    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, str] = {
        "upstream": UPSTREAM,
        "tiny": json.dumps(TINY),
        "dino": json.dumps(DINO),
        "cases": json.dumps(CASES),
        "nvs_targets": json.dumps(NVS_TARGETS),
        "sources": json.dumps(SOURCES),
        "stride": json.dumps(STRIDE),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": PIL.__version__,
    }
    g = torch.Generator().manual_seed(0)

    # process_image, on PNG files as upstream reads them.
    with tempfile.TemporaryDirectory() as tmp:
        for i, (h, w) in enumerate(SOURCES):
            image = torch.randint(0, 256, (h, w, 3), generator=g, dtype=torch.uint8)
            path = Path(tmp) / f"{i}.png"
            Image.fromarray(image.numpy()).save(path)
            processed = (process_image(str(path)) + 1) * 0.5  # as its callers do
            assert processed.shape == (3, CROP, CROP)
            tensors[f"process.{i}.source"] = image.permute(2, 0, 1)
            digest = hashlib.sha256(processed.contiguous().numpy().tobytes())
            tensors[f"process.{i}.sha256"] = torch.frombuffer(
                bytearray(digest.digest()), dtype=torch.uint8
            )
            tensors[f"process.{i}.subsample"] = processed[:, ::STRIDE, ::STRIDE]

    for name, case in CASES.items():
        encoder = build_encoder(A, config, case["voxel_size"])
        meta[f"{name}.checksum"] = json.dumps(checksum(encoder))
        b, v, (h, w) = case["batch"], case["views"], case["shape"]
        images = torch.rand(b, v, 3, h, w, generator=g)
        TOKENS.clear()
        with torch.no_grad():
            out = encoder(images, global_step=0, visualization_dump=None)
        ((tokens, patch_start_idx),) = TOKENS
        meta[f"{name}.patch_start_idx"] = json.dumps(patch_start_idx)
        gs, pose = out.gaussians, out.pred_context_pose
        print(name, "gaussians", tuple(gs.means.shape), "of", b * v * h * w, "pixels")
        prefix = f"{name}."
        tensors |= {
            prefix + "images": images,
            prefix + "means": gs.means,
            prefix + "harmonics": gs.harmonics,
            prefix + "opacities": gs.opacities,
            prefix + "scales": gs.scales,
            prefix + "rotations": gs.rotations,
            prefix + "c2w": pose["extrinsic"],
            prefix + "intrinsics": pose["intrinsic"],
            prefix + "depth": out.depth_dict["depth"],
        }
        tensors |= {f"{prefix}tokens.{i}": t for i, t in enumerate(tokens)}
        if name == "single":
            targets = torch.rand(b, NVS_TARGETS, 3, h, w, generator=g)
            TOKENS.clear()
            with torch.no_grad():
                c2w, k = eval_nvs_target_cameras(encoder, images, targets, pose)
            ((tokens, _),) = TOKENS
            tensors |= {f"{prefix}nvs.tokens.{i}": t for i, t in enumerate(tokens)}
            tensors |= {
                prefix + "nvs.targets": targets,
                prefix + "nvs.c2w": c2w,
                prefix + "nvs.intrinsics": k,
            }
    tensors = {k_: v_.contiguous() for k_, v_ in tensors.items()}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata=meta)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
