"""Freeze outputs of the official TokenGS code for `tests/test_tokengs.py`.

Runs the upstream evaluation data path (`Provider._preprocess`) and model
(`TokenGS.forward_reconstruction`) on tiny inputs and writes them, with the inputs
and the upstream weights, to `tokengs_reference.safetensors`. Needs the official
code importable (https://github.com/nv-tlabs/TokenGS at the commit in UPSTREAM,
e.g. `pip install -e` of a checkout, or PYTHONPATH), and its dependencies:

    python tests/fixtures/make_tokengs_reference.py --out tests/fixtures/tokengs_reference.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchvision
from safetensors.torch import save_file

UPSTREAM = "b16269c500a8894cda342bc9cf406e31169541e3"

# Tiny versions of the two released architectures (checkpoints/ and
# checkpoints_26.06/ latent-bottleneck), same code paths.
TINY = dict(
    patch_size=4,
    dec_patch_size=2,
    enc_depth=2,
    dec_depth=2,
    enc_embed_dim=16,
    enc_num_heads=2,
    num_gs_tokens=4,
    token_dim=16,
    mlp_ratio=1,
)
ARCHS = {
    "base": dict(TINY, gaussian_z_offset=1.0),
    "latent": dict(
        TINY,
        dec_depth=1,
        clip_head_z_init=0.1,
        dec_init_values=0.01,
        gaussian_z_offset=0.0,
        gs_token_std=0.02,
        use_multiscale_encoder=True,
        multiscale_encoder_layers=(0, 1),
        use_latent_bottleneck=True,
        num_latents=6,
        latent_cross_attn_depth=1,
    ),
}
# Source images with odd crop margins in height (28 -> 25) and width (61 -> 48),
# so the upstream half-pixel offset shows.
CASES = {
    "first_cam": dict(arch="base", source=(28, 50), inputs=3, targets=2),
    "mean_cam": dict(arch="latent", source=(24, 61), inputs=2, targets=2),
}
IMAGE_SHAPE = (8, 16)
SCENE_SCALE = 0.15


def random_c2w(g: torch.Generator, n: int) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(n, 3, 3, generator=g, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None]
    q[torch.linalg.det(q) < 0, :, 0] *= -1
    c2w = torch.eye(4, dtype=torch.float64).repeat(n, 1, 1)
    c2w[:, :3, :3] = q
    c2w[:, :3, 3] = 3 * torch.randn(n, 3, generator=g, dtype=torch.float64)
    return c2w.float()  # as upstream: float64 poses -> float32


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import torchvision.transforms as transforms
    from tokengs.data.provider import Provider
    from tokengs.models import TokenGS
    from tokengs.models.input_types import (
        ModelInput,
        ModelInputDecoder,
        ModelInputEncoder,
    )
    from tokengs.options import Options
    from tokengs.utils.data import ImageTransform

    tensors: dict[str, torch.Tensor] = {}
    meta: dict[str, str] = {
        "upstream": UPSTREAM,
        "archs": json.dumps(ARCHS),
        "cases": json.dumps(CASES),
        "image_shape": json.dumps(IMAGE_SHAPE),
        "scene_scale": json.dumps(SCENE_SCALE),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    g = torch.Generator().manual_seed(0)
    for name, case in CASES.items():
        arch = ARCHS[case["arch"]]
        opt = Options(
            img_size=IMAGE_SHAPE,
            num_input_views=case["inputs"],
            camera_normalization_method=name,
            evaluating=True,
            lambda_lpips=0.0,
            **arch,
        )
        views = case["inputs"] + case["targets"]
        h, w = case["source"]
        images = torch.randint(0, 256, (views, 3, h, w), generator=g, dtype=torch.uint8)
        # [fx, fy, cx, cy] in pixels, principal point off centre.
        k = torch.stack(
            [
                torch.full((views,), 0.9 * w),
                torch.full((views,), 0.95 * h),
                w / 2 + torch.randn(views, generator=g),
                h / 2 + torch.randn(views, generator=g),
            ],
            dim=-1,
        ).float()
        c2w = random_c2w(g, views)

        # The evaluation data path, as `Provider.get_item` runs it.
        provider = Provider.__new__(Provider)
        provider.opt = opt
        provider.scene_scale = SCENE_SCALE
        provider.training = False
        provider.image_transform = ImageTransform(
            crop_size=opt.img_size, sample_size=opt.img_size, max_crop=True
        )
        provider.input_normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False
        )
        rgbs = images / 255.0
        ones = torch.ones_like(rgbs[:, :1])
        out = provider._preprocess(
            rgbs, ones, ones, c2w.clone(), k.clone(), None, False
        )

        model = TokenGS(opt).eval()
        torch.manual_seed(1)
        with torch.no_grad():  # non-trivial weights everywhere, incl. LayerScale
            for p in model.parameters():
                p.add_(0.05 * torch.randn_like(p))
        n = opt.num_input_views
        batch = {k_: v[None] for k_, v in out.items() if torch.is_tensor(v)}
        encoder = ModelInputEncoder(
            images_rgb=batch["input"][:, :n, :3],
            plucker=batch["input"][:, :n, -6:],
            rays_os=batch["rays_os"][:, :n],
            rays_ds=batch["rays_ds"][:, :n],
            intrinsics_input=batch["intrinsics_input"],
            cam_to_world_input=batch["cam_to_world_input"],
        )
        with torch.no_grad():
            gaussians = model.forward_reconstruction(
                ModelInput(encoder=encoder, decoder=ModelInputDecoder())
            ).gaussians

        prefix = f"{name}."
        tensors |= {
            prefix + "source.images": images,
            prefix + "source.intrinsics": k,
            prefix + "source.c2w": c2w,
            prefix + "images": out["images_all"],
            prefix + "intrinsics": out["intrinsics_all"],
            prefix + "cam_view": out["cam_view_all"],
            prefix + "model_c2w_input": out["cam_to_world_input"],
            prefix + "input.images": out["input"][:n, :3],
            prefix + "input.plucker": out["input"][:n, -6:],
            prefix + "gaussians": gaussians[0],
        }
        tensors |= {prefix + "weights." + k_: v for k_, v in model.state_dict().items()}
    tensors = {k_: v.contiguous() for k_, v in tensors.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata=meta)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
