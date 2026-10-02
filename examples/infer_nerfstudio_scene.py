"""Reconstruct a nerfstudio-style scene with an ffgs model and render novel views.

Reads a scene in the nerfstudio / DL3DV layout (`transforms.json` with
`fl_x fl_y cx cy w h` and a per-frame OpenGL `transform_matrix`), feeds evenly
spaced frames to the model, writes the Gaussians as a 3DGS ply in the scene's own
frame (that of `transforms.json`) and renders frames in between the inputs
(rendering needs CUDA and `pip install "ffgs[render]"`).

    python examples/infer_nerfstudio_scene.py <scene dir> \\
        [--model tokengs/dl3dv-6v] [--images-dir <scene>/images_4] [--out out]

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

from ffgs import Cameras, GSPipeline, Views

# OpenGL / Blender camera axes (y up, z backward) -> OpenCV (y down, z forward).
GL_TO_CV = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))


def load_scene(
    scene_dir: Path, images_dir: Path | None = None
) -> tuple[list[Path], torch.Tensor, torch.Tensor]:
    """Image paths, normalised intrinsics [N, 3, 3] and OpenCV c2w [N, 4, 4],
    frames sorted by file name. Per-frame intrinsics override the global ones.
    With `images_dir`, images are read from there by file name (e.g. a
    downsampled copy: normalised intrinsics hold at any resolution)."""
    meta = json.loads((scene_dir / "transforms.json").read_text())
    frames = sorted(meta["frames"], key=lambda f: f["file_path"])
    paths, ks, c2w = [], [], []
    for frame in frames:
        cam = {**meta, **frame}
        k = torch.tensor(
            [[cam["fl_x"], 0, cam["cx"]], [0, cam["fl_y"], cam["cy"]], [0, 0, 1]],
            dtype=torch.float64,
        )
        k[0] /= cam["w"]
        k[1] /= cam["h"]
        ks.append(k.float())
        path = Path(frame["file_path"])
        paths.append(images_dir / path.name if images_dir else scene_dir / path)
        matrix = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
        c2w.append(matrix @ GL_TO_CV)
    return paths, torch.stack(ks), torch.stack(c2w)


def read_rgb(path: Path) -> torch.Tensor:
    """uint8 [3, H, W]."""
    with Image.open(path) as image:
        return torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1)


def evenly_spaced(n: int, count: int) -> torch.Tensor:
    return torch.linspace(0, n - 1, count).round().long().unique()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("scene_dir", type=Path, help="directory with transforms.json")
    parser.add_argument("--model", default="tokengs/dl3dv-6v")
    parser.add_argument("--images-dir", type=Path, help="read images from here")
    parser.add_argument(
        "--num-views", type=int, help="input views (default: the model's, else 4)"
    )
    parser.add_argument("--num-render", type=int, default=4, help="0: no rendering")
    parser.add_argument("--scene-scale", type=float, help="override the model's")
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = GSPipeline.from_pretrained(args.model, device=device)
    num_views = (
        args.num_views or getattr(pipe.processor.config, "num_input_views", None) or 4
    )

    paths, k, c2w = load_scene(args.scene_dir, args.images_dir)
    context = evenly_spaced(len(paths), num_views)
    images = torch.stack([read_rgb(paths[i]) for i in context])
    views = Views(images, k[context], c2w[context], normalized_intrinsics=True)
    options = {} if args.scene_scale is None else {"scene_scale": args.scene_scale}
    out = pipe(views, **options)

    args.out.mkdir(parents=True, exist_ok=True)
    # `out` holds the Gaussians in the model's frame; the ply goes in the scene's.
    ply = out.to_world().gaussians.save_ply(args.out / "gaussians.ply")
    print(f"{out.gaussians.num_gaussians} Gaussians -> {ply}")
    if args.num_render <= 0:
        return

    # Frames halfway between consecutive inputs, rendered at the model's input
    # size with the inputs' resize + crop (so they line up with the inputs). The
    # cameras are in the scene's frame; `render` takes them into the model's.
    middle = ((context[:-1] + context[1:]) // 2).unique()
    if len(middle) == 0:
        print("one input view: no frames between inputs to render")
        return
    middle = middle[evenly_spaced(len(middle), args.num_render)]
    source = Cameras(c2w[middle], k[middle], views.image_shape, True)
    rendered = pipe.render(out, pipe.processor.fit_cameras(source))["images"][0]
    for index, image in zip(middle.tolist(), rendered):
        path = args.out / f"render_{index:05d}.png"
        pixels = (image.clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0)
        Image.fromarray(pixels.cpu().numpy()).save(path)
        print(f"frame {index} -> {path}")


if __name__ == "__main__":
    main()
