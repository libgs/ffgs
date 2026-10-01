"""Reconstruct a folder of unposed images with a model that predicts its cameras.

Reads the images of a folder (sorted by file name, all the same size), feeds
evenly spaced ones to the model without intrinsics or poses, and writes, in the
model's own frame (there is no other: no poses went in):

- `gaussians.ply`, a standard 3DGS ply;
- `cameras.json`, the predicted camera of each input image (OpenCV c2w and
  intrinsics normalised by the image size, for the model's input size);
- `render_<index>.png`, the inputs re-rendered at their predicted cameras
  (rendering needs CUDA and `pip install "ffgs[render]"`).

    python examples/infer_pose_free.py <image dir> \\
        [--model anysplat/default] [--num-views 8] [--device cuda] [--out out]

`--model` is anything `GSPipeline.from_pretrained` takes: a zoo name, a directory
written by `save_pretrained` or a Hub repo id. Check the license of the weights
before use (`ffgs.zoo` entries carry it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ffgs import GSPipeline, Views

SUFFIXES = (".png", ".jpg", ".jpeg")


def image_paths(folder: Path) -> list[Path]:
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUFFIXES)
    if not paths:
        raise SystemExit(f"no {'/'.join(SUFFIXES)} images in {folder}")
    return paths


def read_rgb(path: Path) -> torch.Tensor:
    """uint8 [3, H, W]."""
    with Image.open(path) as image:
        return torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)


def evenly_spaced(n: int, count: int) -> torch.Tensor:
    return torch.linspace(0, n - 1, min(count, n)).round().long().unique()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--model", default="anysplat/default")
    parser.add_argument("--num-views", type=int, default=8, help="input images")
    parser.add_argument(
        "--no-render", action="store_true", help="skip the renders (CPU only)"
    )
    parser.add_argument("--device", help="default: cuda if available, else cpu")
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipe = GSPipeline.from_pretrained(args.model, device=device)

    paths = image_paths(args.image_dir)
    picked = evenly_spaced(len(paths), args.num_views).tolist()
    images = torch.stack([read_rgb(paths[i]) for i in picked])
    out = pipe(Views(images))  # no poses: out.frame is None, all in the model's frame

    args.out.mkdir(parents=True, exist_ok=True)
    ply = out.gaussians.save_ply(args.out / "gaussians.ply")
    print(f"{out.gaussians.num_gaussians} Gaussians -> {ply}")
    cameras = out.cameras
    record = {
        "frame": "model",
        "image_shape": list(cameras.image_shape),
        "views": [
            {
                "index": index,
                "file": paths[index].name,
                "c2w": cameras.c2w[0, v].tolist(),
                "k_normalized": cameras.normalized_k[0, v].tolist(),
            }
            for v, index in enumerate(picked)
        ],
    }
    (args.out / "cameras.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"{len(picked)} predicted cameras -> {args.out / 'cameras.json'}")
    if args.no_render:
        return

    # The predicted cameras are in the model's frame, like the Gaussians.
    rendered = pipe.render(out, cameras)["images"][0]
    for index, image in zip(picked, rendered):
        path = args.out / f"render_{index:05d}.png"
        pixels = (image.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
        Image.fromarray(pixels.cpu().numpy()).save(path)
        print(f"image {index} -> {path}")


if __name__ == "__main__":
    main()
