"""Reconstruct a nerfstudio-style scene with an ffgs model and render novel views.

Reads a scene in the nerfstudio / DL3DV layout (`images/` and their cameras in
`transforms.json`), feeds evenly spaced images (by file name) to the model,
writes the Gaussians as a 3DGS ply in the scene's own frame (that of
`transforms.json`) and renders the images in between the inputs (rendering needs
CUDA and `pip install "ffgs[render]"`).

    python examples/infer_nerfstudio_scene.py <scene dir> \\
        [--model tokengs/dl3dv-6v] [--images-dir <scene>/images_4] [--out out]

`--model` is anything `GSPipeline.from_pretrained` takes: a zoo name, a directory
written by `save_pretrained` or a Hub repo id. Check the license of the weights
before use (`ffgs.zoo` entries carry it).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from ffgs import Cameras, GSPipeline, Views
from ffgs.io import evenly_spaced, image_paths, pair_cameras, read_cameras, read_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scene_dir", type=Path, help="directory with transforms.json")
    parser.add_argument("--model", default="tokengs/dl3dv-6v")
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="read images from here (default: <scene>/images)",
    )
    parser.add_argument("--num-views", type=int, default=4, help="input views")
    parser.add_argument("--num-render", type=int, default=4, help="0: no rendering")
    parser.add_argument("--scene-scale", type=float, help="override the model's")
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = GSPipeline.from_pretrained(args.model, device=device)

    transforms = args.scene_dir / "transforms.json"
    paths = image_paths(args.images_dir or args.scene_dir / "images")
    context = torch.tensor(evenly_spaced(len(paths), args.num_views))
    cameras = pair_cameras(paths, read_cameras(transforms))  # by file name
    inputs = [cameras[i] for i in context]
    views = Views(
        torch.stack([read_image(paths[i]) for i in context]),  # all the same size
        torch.stack([camera.k for camera in inputs]),
        torch.stack([camera.c2w for camera in inputs]),
        normalized_intrinsics=True,
    )
    options = {} if args.scene_scale is None else {"scene_scale": args.scene_scale}
    out = pipe(views, **options)

    args.out.mkdir(parents=True, exist_ok=True)
    # `out` holds the Gaussians in the model's frame; the ply goes in the scene's.
    ply = out.to_world().save_ply(args.out / "gaussians.ply")
    print(f"{out.gaussians.num_gaussians} Gaussians -> {ply}")
    if args.num_render <= 0:
        return

    # Images halfway between consecutive inputs, rendered at the model's input
    # size with the inputs' resize + crop (so they line up with the inputs). The
    # cameras are in the scene's frame; `render` takes them into the model's.
    middle = ((context[:-1] + context[1:]) // 2).unique()
    if len(middle) == 0:
        print("one input view: no frames between inputs to render")
        return
    middle = middle[evenly_spaced(len(middle), args.num_render)]
    between = [cameras[i] for i in middle]
    k = torch.stack([camera.k for camera in between])
    c2w = torch.stack([camera.c2w for camera in between])
    source = Cameras(c2w, k, views.image_shape, True)
    rendered = pipe.render(out, pipe.processor.fit_cameras(source))["images"][0]
    for index, image in zip(middle.tolist(), rendered):
        path = args.out / f"render_{index:05d}.png"
        pixels = (image.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
        Image.fromarray(pixels.cpu().numpy()).save(path)
        print(f"{paths[index].name} -> {path}")


if __name__ == "__main__":
    main()
