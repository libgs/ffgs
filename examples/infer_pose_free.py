"""Reconstruct a folder of unposed images with a model that predicts its cameras.

Reads the images of a folder (sorted by file name, all the same size), feeds
evenly spaced ones to the model without intrinsics or poses, and writes, in the
model's own frame (there is no other: no poses went in):

- `gaussians.ply`, a standard 3DGS ply;
- `cameras.json`, the cameras the model predicts, as they are (see
  `ffgs.io.save_cameras`): one per input image, in order, named by position;
- `render_<i>.png`, the inputs re-rendered at their predicted cameras
  (rendering needs CUDA and `pip install "ffgs[render]"`).

    python examples/infer_pose_free.py <image dir> \\
        [--model anysplat/default] [--num-views 8] [--device cuda] [--out out]

`--model` is anything `GSPipeline.from_pretrained` takes: a zoo name, a directory
written by `save_pretrained` or a Hub repo id. Check the license of the weights
before use (`ffgs.zoo` entries carry it).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from ffgs import GSPipeline, Views
from ffgs.io import Camera, evenly_spaced, image_paths, read_image, save_cameras


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
    paths = [paths[i] for i in evenly_spaced(len(paths), args.num_views)]
    images = torch.stack([read_image(path) for path in paths])  # all the same size
    out = pipe(Views(images))  # no poses: out.frame is None, all in the model's frame

    args.out.mkdir(parents=True, exist_ok=True)
    ply = out.save_ply(args.out / "gaussians.ply")
    print(f"{out.gaussians.num_gaussians} Gaussians -> {ply}")
    h, w = out.cameras.image_shape
    k, c2w = out.cameras.normalized_k[0].cpu(), out.cameras.c2w[0].cpu()
    cameras = [Camera(str(i), k[i], c2w[i], (w, h)) for i in range(len(paths))]
    saved = save_cameras(args.out / "cameras.json", cameras)
    print(f"{len(cameras)} predicted cameras -> {saved}")
    if args.no_render:
        return

    # The predicted cameras are in the model's frame, like the Gaussians.
    rendered = pipe.render(out, out.cameras)["images"][0]
    for i, (image, source) in enumerate(zip(rendered, paths)):
        path = args.out / f"render_{i}.png"
        pixels = (image.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
        Image.fromarray(pixels.cpu().numpy()).save(path)
        print(f"{source.name} -> {path}")


if __name__ == "__main__":
    main()
