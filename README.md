# ffgs — standard inference for feed-forward 3D Gaussian Splatting

One way to load and run feed-forward 3D Gaussian Splatting models: posed images in,
Gaussians out — in the model's own frame, or in **your** world frame with one call —
whatever model is behind it.

> Early development: the core interface, third-party loading and the model zoo
> are in place, with TokenGS as the first model (reproduced exactly, see
> [Results](#results)); more models are being added.

```python
from ffgs import Cameras, GSPipeline, Views

pipe = GSPipeline.from_pretrained("tokengs/dl3dv-6v", device="cuda")  # or a path / Hub repo
out = pipe(Views(images, intrinsics, c2w))          # -> GSOutput, in the model's frame
images = pipe.render(out, Cameras(c2w_new, k_new, (h, w)))["images"]  # cameras in your frame
out.to_world().gaussians.save_ply("scene.ply")      # standard 3DGS ply, in the frame of c2w
```

## Install

```bash
pip install ffgs            # inference
pip install "ffgs[render]"  # + gsplat rasterisation (CUDA)
```

Dependencies: torch, torchvision, numpy, pillow, huggingface_hub, safetensors;
gsplat (the `render` extra) is imported only when rendering.

## Example

[`examples/infer_nerfstudio_scene.py`](https://github.com/libgs/ffgs/blob/main/examples/infer_nerfstudio_scene.py)
reconstructs a scene in the nerfstudio / DL3DV layout (`transforms.json`, OpenGL
camera poses) from evenly spaced frames, writes a 3DGS ply in the scene's frame and
renders the frames between the inputs:

```bash
python examples/infer_nerfstudio_scene.py <scene> --model tokengs/dl3dv-6v \
    --images-dir <scene>/images_4 --out out
```

## Models

| zoo name | model | input views | input size | weights |
|---|---|---|---|---|
| `tokengs/dl3dv-base` | TokenGS, 1024 Gaussian tokens (the base the variants below are finetuned from) | 4 | 256×256 | [jiaweir/tokengs](https://huggingface.co/jiaweir/tokengs) |
| `tokengs/dl3dv-{2,4,6}v` | TokenGS, 4096 Gaussian tokens | 2 / 4 / 6 | 256×448 | [jiaweir/tokengs](https://huggingface.co/jiaweir/tokengs) |
| `tokengs/dl3dv-latent-{2,4,6}v-{ssim,lpips}` | TokenGS 2026.6 latent-bottleneck models, trained with an SSIM or an LPIPS loss | 2 / 4 / 6 | 256×448 | [jiaweir/tokengs](https://huggingface.co/jiaweir/tokengs) |

TokenGS ([paper](https://arxiv.org/abs/2604.15239),
[code](https://github.com/nv-tlabs/TokenGS)) is trained on DL3DV. Its code is
vendored under Apache-2.0 (`src/ffgs/models/tokengs/`, feed-forward inference only:
no test-time training); **the weights are under the NVIDIA Internal Scientific
Research and Development Model License (non-commercial)**. `ffgs.zoo` entries list
the source, the pinned revision and the license of each checkpoint.

## Results

TokenGS on the DL3DV benchmark (140 scenes of
[DL3DV-10K-Benchmark](https://huggingface.co/datasets/DL3DV/DL3DV-10K-Benchmark),
the upstream evaluation index and target views, no test-time training), run with
the authors' evaluation code and with ffgs on the same scenes, frames and
checkpoints. For all 11 configurations below, **the Gaussians from ffgs are
bit-identical to those of the official code in 140/140 scenes**, so both report the
same means; per scene, the metrics differ by at most 4.2e-4 PSNR, 7.9e-6 SSIM and
6.6e-5 LPIPS, from rendering (these ffgs runs rendered the Gaussians taken to the
caller's world frame, the official code in the model frame; rendering the official
Gaussians with ffgs gives the ffgs images exactly).

| zoo entry | views | PSNR ↑ | SSIM ↑ | LPIPS ↓ | published | Δ PSNR / SSIM / LPIPS |
|---|---|---|---|---|---|---|
| `tokengs/dl3dv-4v` | 2 | 19.382 | 0.598 | 0.438 | 19.35 / 0.609 / 0.440 | +0.03 / −0.011 / −0.002 |
| `tokengs/dl3dv-4v` | 4 | 23.048 | 0.740 | 0.325 | 23.02 / 0.747 / 0.326 | +0.03 / −0.007 / −0.001 |
| `tokengs/dl3dv-4v` | 6 | 23.715 | 0.759 | 0.310 | 23.69 / 0.766 / 0.311 | +0.03 / −0.007 / −0.001 |
| `tokengs/dl3dv-2v` | 2 | 20.181 | 0.635 | 0.420 | — | |
| `tokengs/dl3dv-6v` | 6 | 23.892 | 0.766 | 0.306 | — | |
| `tokengs/dl3dv-latent-2v-ssim` | 2 | 20.540 | 0.659 | 0.380 | 20.493 / 0.658 / 0.381 | +0.05 / +0.001 / −0.001 |
| `tokengs/dl3dv-latent-4v-ssim` | 4 | 24.066 | 0.779 | 0.277 | 24.012 / 0.778 / 0.277 | +0.05 / +0.001 / 0.000 |
| `tokengs/dl3dv-latent-6v-ssim` | 6 | 25.170 | 0.806 | 0.258 | 25.120 / 0.806 / 0.257 | +0.05 / 0.000 / +0.001 |
| `tokengs/dl3dv-latent-2v-lpips` | 2 | 19.840 | 0.623 | 0.317 | 19.772 / 0.620 / 0.317 | +0.07 / +0.003 / 0.000 |
| `tokengs/dl3dv-latent-4v-lpips` | 4 | 23.171 | 0.756 | 0.216 | 23.089 / 0.754 / 0.215 | +0.08 / +0.002 / +0.001 |
| `tokengs/dl3dv-latent-6v-lpips` | 6 | 24.152 | 0.785 | 0.194 | 24.077 / 0.783 / 0.194 | +0.08 / +0.002 / 0.000 |

Published: `tokengs/dl3dv-4v` rows from Table 2 of the paper (the 4-view model
evaluated with 2, 4 and 6 input views), latent rows from the authors' 2026.6
release notes; the finetuned 2- and 6-view models have no published numbers. Notes:

- The gaps to the published numbers are the same for the official code and ffgs,
  so they are not from ffgs. Likely causes, not isolated: torch 2.7.1 / numpy 2.2.6
  instead of the upstream pins (torch 2.7.0, numpy < 2), and bf16 inference, which
  is not bit-reproducible across GPU architectures (these runs used an RTX PRO 6000
  Blackwell; the same scene on an RTX 4090 moves PSNR by a few hundredths). The
  SSIM of the paper rows is 0.007–0.011 below Table 2 with both; we have not traced
  why.
- The unmodified upstream `evaluate.py` matches the benchmark's official-side
  script within 5e-5 on the paper and finetuned rows. It cannot run the
  `latent-*-lpips` presets in this setup: it also computes the training loss,
  whose LPIPS term fails under bf16 autocast, so the benchmark disables that
  (discarded) loss term.

The scripts (environment setup, data packing, both evaluations, the scene-by-scene
comparison) are on the
[`bench/tokengs-dl3dv`](https://github.com/libgs/ffgs/tree/bench/tokengs-dl3dv/benchmarks/tokengs_dl3dv)
branch.

## Conventions

| | |
|---|---|
| images | float in [0, 1] (uint8 accepted), `[V, 3, H, W]` or `[B, V, 3, H, W]`, any resolution |
| c2w | camera-to-world, OpenCV axes (x right, y down, z forward); optional for models that predict their own cameras |
| intrinsics | pixel K for the image size given, or normalised K with `normalized_intrinsics=True`; optional like c2w |
| output | `GSOutput`: `Gaussians` (means, scales, wxyz quats, opacities, SH or RGB colours) in the model's frame, that frame (or `None`), and the input cameras the model predicted (or `None`) |

A processor reproduces its model's training preprocessing: rescale to cover the
model's input size, centre crop (normalised principal point kept, as in training),
cameras relative to the first input view, translations times a model-specific
`scene_scale`. The model predicts in that normalised frame, and that is where
`pipe(views)` leaves the Gaussians (`out.space == "model"`), exactly as the
model's own code produces them.

| | Gaussians in |
|---|---|
| `out = pipe(views)` | the model's frame (`out.frame`: world → model) |
| `out.to_world()` | the world frame of the input c2w (fp32) |
| `out.gaussians.save_ply(path)` | whichever frame `out` is in: `out.to_world().gaussians.save_ply(path)` for a ply that lines up with your cameras |
| `pipe.render(out, cameras)` | either: `cameras` are always in the world frame of the input c2w, and are taken (with near / far) into the model's frame when needed |

`to_world()` is one similarity transform for every model — rotation, uniform scale
and translation — applied per attribute: means fully, scales by the scale, quats by
the rotation, view-dependent SH colours of any degree by an exact rotation of their
coefficients (3DGS / gsplat basis); opacities and RGB / degree-0 colours are
unchanged. Input c2w whose rotation is not orthonormal or is a reflection (det < 0)
are rejected.

Models that need poses (TokenGS) raise a `ValueError` naming what is missing.
Models that predict their own cameras accept `Views(images)` alone, and
`out.cameras` holds their prediction for the input views, in the same frame as
`out.gaussians` (`to_world()` moves both):

| views | `out.frame` | `to_world()` | `pipe.render(out, cameras)` |
|---|---|---|---|
| without c2w | `None`: there is no world frame | raises | `cameras` in the model's frame, e.g. `out.cameras`; near / far in model units |
| with c2w (V ≥ 2) | fitted to the given c2w | as above | as above: `cameras` in the world frame of the input c2w |

With c2w, the frame is one closed-form fit over all input views of the predicted
cameras onto the given ones: the rotation is the mean of the per-view rotations
(SVD, det +1), then scale and translation are the least-squares fit of the camera
centres. The given c2w do not enter the model. `out.frame.residuals` reports how
well they agree, per view: rotation error in degrees, centre error in world
units, and centre error relative to the RMS spread of the given centres. Fewer
than 2 views, coincident predicted centres or a fit with negative scale raise.

Resize + crop is the default, not a requirement. Processor options (in
`processor_config.json` or `processor_overrides=`):

| option | |
|---|---|
| `crop_mode="crop"` | cover the input shape, centre crop — the training rule (default) |
| `crop_mode="pad"` | fit inside the input shape and pad with `pad_value`; nothing is cut off, K follows the content |
| `crop_mode="none"` | inputs are already at the model's shape; used as they are |
| `resize_mode` | built-in kernel: `"bilinear"` (antialiased) or `"lanczos"` |
| `resize=fn` | your own `fn(images, (h, w)) -> images`, replacing `resize_mode` (not serialised) |

```python
pipe = GSPipeline.from_pretrained(path, processor_overrides={"crop_mode": "pad"})
```

## Layers

| | |
|---|---|
| `types` | `Views`, `Cameras`, `Gaussians` (+ `save_ply`) |
| `processor` | per-model pre / post-processing, stored as `processor_config.json` |
| model | the model's own `nn.Module` with `PyTorchModelHubMixin` (`config.json` + safetensors) |
| `pipeline` | `GSPipeline`: `from_pretrained` dispatches on `config.json:model_type` |
| `render` | gsplat rasterisation of `Gaussians` at `Cameras` (gsplat imported on first use) |
| `zoo` | named checkpoints: `zoo/<family>/<variant>.json` pointers to pinned upstream weights |

A saved model is a directory (or Hub repo) with `config.json` (model config +
`model_type`), `model.safetensors` and `processor_config.json`;
`GSPipeline.save_pretrained` writes all three.

## Model zoo

The zoo names published checkpoints: one JSON file per variant in
`src/ffgs/zoo/<family>/<variant>.json`, shipped with the package. The weights stay
in their authors' Hub repos; an entry pins the file to a commit and a sha256.

```python
from ffgs import GSPipeline, zoo

zoo.list_models()                        # ["<family>/<variant>", ...]
pipe = GSPipeline.from_pretrained("<family>/<variant>", device="cuda")
```

```json
{
  "model_type": "my-model",
  "description": "My model trained on X, 2 input views",
  "weights": {"repo": "author/repo", "file": "ckpt/model.pt",
              "revision": "<full 40-hex commit>", "sha256": "<64-hex>"},
  "model": {"...": "model constructor arguments"},
  "processor": {"...": "processor_config.json"},
  "source": {"url": "https://github.com/author/code", "paper": "https://..."},
  "license": "license of the weights"
}
```

Loading an entry downloads the file at the pinned commit, fails if its sha256
differs, converts the upstream checkpoint with the model's
`ModelSpec.convert_state_dict` (e.g. stripping a key prefix) and loads it strictly:
every key must match. `.pt` checkpoints are unpickled with `weights_only=True`.
`save_pretrained` then writes an ordinary ffgs directory.

`from_pretrained(name)` tries, in order: an existing local directory, a zoo entry,
a Hub repo id. A zoo family name is reserved — `"<family>/<anything>"` never goes
to the Hub, and an unknown variant is an error listing the known ones. (To load a
Hub repo whose owner shares a family name, `huggingface_hub.snapshot_download` it
and pass the directory.) A zoo entry pins its own revision, so `revision=` and
`subfolder=` are rejected for it; `cache_dir`, `token`, `force_download` and
`local_files_only` apply.

Checkpoints saved with `save_pretrained` can also be published as Hub repos, one
per model family with variants in subfolders:

```python
pipe = GSPipeline.from_pretrained("owner/family-repo", subfolder="variant",
                                  revision="<commit sha>")
```

## Adding a model

A model is a model class with `PyTorchModelHubMixin`, a `Processor` subclass
(`preprocess`, returning the model input and its `ModelFrame`, or `None` when the
model predicts its own cameras; `postprocess`, to model-frame `Gaussians`;
`render_planes`, near / far in world units, or in model units when the frame is
`None`; for models that predict cameras, `predicted_cameras`, from the model output
to model-frame `Cameras`, and `output_frame`, whose default fits the frame to the
given c2w as above) and a
`ModelSpec(model_type, model_cls, processor_cls)`. `from_pretrained` resolves the
`model_type` in `config.json` in this order:

1. **In your code** — `ffgs.register_model(spec)` before loading (built-in models
   are registered the same way, on first use).
2. **An installed package** — expose the spec under the `ffgs.models` entry point
   group, named by its `model_type`; ffgs imports the package only when that
   `model_type` is loaded:

   ```toml
   # pyproject.toml of your package
   [project.entry-points."ffgs.models"]
   my-model = "my_package.ffgs_model:SPEC"   # a ModelSpec with model_type "my-model"
   ```

3. **Code in the model repo** — ship the `.py` files next to `config.json` and name
   the classes (relative imports between the files work):

   ```json
   {"model_type": "my-model",
    "auto_map": {"model": "modeling_my.MyModel", "processor": "processing_my.MyProcessor"}}
   ```

   That code runs only with `trust_remote_code=True` (off by default); read it
   first and pin the revision:

   ```python
   pipe = GSPipeline.from_pretrained("someone/my-model", trust_remote_code=True,
                                     revision="<commit sha>")
   ```

   `save_pretrained` on such a pipeline copies the code and keeps `auto_map`.
   To publish a model whose code lives in your own (e.g. training) codebase, keep
   the model and processor classes in one package that imports itself only
   relatively, and export it with the weights:

   ```python
   GSPipeline(model, processor, model_type="my-model").save_pretrained(
       "export/my-model", include_code=True)   # copies the package, writes auto_map
   ```

   Imports of the rest of your codebase are refused at export time, so the
   directory loads with `trust_remote_code=True` where your codebase is not
   installed.

An installed model wins over repo code with the same `model_type`. Hub options
(`revision`, `subfolder`, `cache_dir`, `token`, `force_download`,
`local_files_only`) apply to every file `from_pretrained` downloads.

To add a zoo entry for a model, write its JSON (compute the sha256 of the pinned
file), give the model's `ModelSpec` a `convert_state_dict` if the upstream keys
differ from the model's, and run the tests: every shipped entry is checked to
parse and build offline, and `uv run pytest --network` downloads each one and
loads it strictly.

## Development

```bash
uv sync
uv run pytest            # everything; tests marked `gpu` skip without CUDA + gsplat
uv run pytest -m gpu     # only the GPU tests (needs CUDA and `--extra render`)
uv run pytest --network  # also the tests that download zoo weights from the Hub
uv run pre-commit install
```

`uv sync` installs torch from PyPI (CUDA build). CI has no GPU and installs CPU
torch first (see `.github/workflows/ci.yml`). `tests/fixtures/reference.safetensors`
holds frozen reference outputs of the default data path; change it only together
with a deliberate change to that path.

## License

Apache-2.0 (see `LICENSE`). `src/ffgs/image.py` is adapted from pixelSplat (MIT);
its license is included at the end of `LICENSE`.
