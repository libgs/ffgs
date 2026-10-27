# ffgs — standard inference for feed-forward 3D Gaussian Splatting

One way to load and run feed-forward 3D Gaussian Splatting models: posed images in,
Gaussians out — in the model's own frame, or in **your** world frame with one call —
whatever model is behind it.

> Early development: the core interface, third-party loading and the model zoo
> are in place; see
> [the models](https://github.com/libgs/ffgs/blob/main/docs/models/index.md)
> for those shipped so far.

```python
from ffgs import Cameras, GSPipeline, Views

pipe = GSPipeline.from_pretrained("<family>/<variant>", device="cuda")  # or a path / Hub repo
out = pipe(Views(images, intrinsics, c2w))          # -> GSOutput, in the model's frame
images = pipe.render(out, Cameras(c2w_new, k_new, (h, w)))["images"]  # cameras in your frame
out.to_world().save_ply("scene.ply")                # standard 3DGS ply, in the frame of c2w
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
python examples/infer_nerfstudio_scene.py <scene> --model <family>/<variant> \
    --images-dir <scene>/images_4 --out out
```

Models that do not need poses (`camera_input` "optional" or "none") also run on
images alone. There is no world frame then: the Gaussians, and the input cameras
for models that predict them, are in the model's frame, and renders take cameras
in that frame (see [Conventions](#conventions)):

```python
pipe = GSPipeline.from_pretrained("<family>/<variant>", device="cuda")  # e.g. anysplat/default
out = pipe(Views(images))                    # no intrinsics, no c2w: out.frame is None
out.save_ply("scene.ply")                    # in the model's frame
renders = pipe.render(out, out.cameras)["images"]  # the inputs, at the predicted cameras
```

[`examples/infer_pose_free.py`](https://github.com/libgs/ffgs/blob/main/examples/infer_pose_free.py)
does this for a folder of images: it writes the ply, the predicted cameras
(`cameras.json`, see [Cameras](#cameras)) and the inputs re-rendered at them:

```bash
python examples/infer_pose_free.py <image dir> --model <family>/<variant> \
    --num-views 8 --out out
```

## Command line

`pip install ffgs` also installs an `ffgs` command, to keep track of models and run
them on a folder of images:

```bash
ffgs ls --zoo                        # the model zoo, and which entries are downloaded
ffgs add <family>/<variant>          # record a model, without downloading it
ffgs pull <family>/<variant>         # download it now (else on first use)
ffgs run <family>/<variant> <images> -o out --cameras <scene>/transforms.json
ffgs run <family>/<variant> <images> -o out --cameras <scene>/sparse/0   # COLMAP
ffgs run <family>/<variant> <images> -o out   # models that need no cameras
ffgs ls                              # the recorded models, downloaded or not
ffgs rm <family>/<variant>           # forget it (its files stay)
```

A source is anything `GSPipeline.from_pretrained` takes: a zoo name, a Hub repo id
(`--revision`, `--subfolder`) or a local directory. `run` reads the images of the
folder (sorted by file name, `--num-views` of them evenly spaced; default 8), all
the same size, with their cameras from `--cameras` (any camera file of
[Cameras](#cameras); each image must have its camera's aspect ratio), and writes
`scene.ply` (in the frame of the cameras, else in the model's) and, for models
that predict cameras, those cameras as `cameras.json`, as they are.

ffgs keeps one small JSON record per model in `$FFGS_HOME/models`
(default `~/.cache/ffgs`); the weights stay in the Hugging Face cache
(`HF_HUB_CACHE`), shared with other tools. A record says where they are, not
what is there: `ffgs ls` checks that each time, so a model whose files were
deleted (e.g. with `hf cache rm`) shows as `not cached` and is downloaded again
when next used. Every model `GSPipeline.from_pretrained` loads is recorded, from
the command line or from Python. `ffgs rm` deletes records only: the weights are
cleaned with the Hugging Face tools (`hf cache rm`), and a local directory is
yours to delete.

Downloads follow the usual Hugging Face settings: `HF_ENDPOINT` for a mirror,
`HF_HUB_OFFLINE=1` to use only what is cached, `HF_TOKEN` for gated repos.

## Models

The shipped models, their zoo names, licenses, usage notes and results against
their official code are in
[docs/models](https://github.com/libgs/ffgs/blob/main/docs/models/index.md),
one page per model.
Check the license of a model before use: some are for non-commercial use only.

## Conventions

| | |
|---|---|
| images | float in [0, 1] (uint8 accepted), `[V, 3, H, W]` or `[B, V, 3, H, W]`, any resolution |
| c2w | camera-to-world, OpenCV axes (x right, y down, z forward); optional unless the model requires them (`camera_input="required"`) |
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
| `out.save_ply(path)` | whichever frame `out` is in: `out.to_world().save_ply(path)` for a ply that lines up with your cameras |
| `pipe.render(out, cameras)` | either: `cameras` are always in the world frame of the input c2w, and are taken (with near / far) into the model's frame when needed |

`to_world()` is one similarity transform for every model — rotation, uniform scale
and translation — applied per attribute: means fully, scales by the scale, quats by
the rotation, view-dependent SH colours of any degree by an exact rotation of their
coefficients (3DGS / gsplat basis); opacities and RGB / degree-0 colours are
unchanged. Input c2w whose rotation is not orthonormal or is a reflection (det < 0)
are rejected.

Models that need poses raise a `ValueError` naming what is missing.
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

`out.save_ply` writes a standard 3DGS ply (INRIA layout) whose header also says
where it came from, one `comment ffgs-<key>: <value>` line per field; other
readers skip comments, and `ffgs.read_ply_meta(path)` reads them back:

```
comment ffgs-version: 0.1.0
comment ffgs-model: anysplat/default
comment ffgs-space: world
comment ffgs-render: classic
comment ffgs-background: 1 1 1
```

| field | |
|---|---|
| `version` | the ffgs version that wrote the file |
| `model` | the zoo name or Hub repo id (`<repo>[/<subfolder>][@<revision>]`) the pipeline was loaded from; left out for a local directory |
| `space` | `world` (after `to_world()`) or `model` |
| `render` | the rasterize mode the model renders with: `classic` or `antialiased` |
| `background` | the model's background colour, `r g b` in [0, 1]; left out when it defines none |

Nothing else goes in: no arrays, paths or timestamps, so the same output always
writes the same file. `Gaussians.save_ply` alone writes no fields.

Resize + crop is the default, not a requirement. Processor options (in
`processor_config.json` or `processor_overrides=`):

| option | |
|---|---|
| `crop_mode="crop"` | cover the input shape, centre crop — the training rule (default) |
| `crop_mode="pad"` | fit inside the input shape and pad with `pad_value`; nothing is cut off, K follows the content |
| `crop_mode="none"` | inputs are already at the model's shape; used as they are |
| `resize_mode` | built-in kernel: `"bilinear"` (antialiased) or `"lanczos"`; a model's processor may add its own (see its page) |
| `resize=fn` | your own `fn(images, (h, w)) -> images`, replacing `resize_mode` (not serialised) |

```python
pipe = GSPipeline.from_pretrained(path, processor_overrides={"crop_mode": "pad"})
```


## Cameras

`ffgs.io` reads and writes camera files; it knows nothing of `Views` or
`Cameras`, so you build those from what it reads. A camera file is a list of
`ffgs.io.Camera`, in order, one per image:

| | |
|---|---|
| `name` | the image's file name; the camera's position (`"0"`, `"1"`, ...) when its source names none |
| `k` | pinhole intrinsics `[3, 3]`, normalised by `size` (fx / w, cx / w, fy / h, cy / h) |
| `c2w` | camera-to-world `[4, 4]`, OpenCV axes |
| `size` | the `(w, h)` the camera is for |

`io.read_cameras(path)` reads them, in the file's order, from

- a nerfstudio / instant-ngp `transforms.json` (`fl_x` / `fl_y` or
  `camera_angle_x` / `camera_angle_y`, `cx` / `cy`, `w` / `h`, globally or per
  frame; OpenGL `transform_matrix`);
- a COLMAP model (`cameras` + `images`, `.bin` or `.txt`; a directory, or the
  files);
- an ffgs `cameras.json`.

Lens distortion is ignored, with a warning: a camera is its pinhole `k` (undistort
the images yourself if it matters). A camera without pinhole intrinsics (e.g.
nerfstudio `EQUIRECTANGULAR`) is an error. `io.pair_cameras(paths, cameras)`
gives each image its camera: by file name, or in order when the cameras are
named by position.

`io.save_cameras(path, cameras)` writes them as an ffgs `cameras.json`, which
`read_cameras` reads back as they were:

```json
[{"name": "0001.png", "k": [[0.8, 0, 0.5], [0, 1.1, 0.5], [0, 0, 1]],
  "c2w": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
  "size": [640, 480]}]
```

Other formats are yours to convert: build the cameras (or write a `cameras.json`).

```python
import torch
from ffgs import Views
from ffgs.io import image_paths, pair_cameras, read_cameras, read_image

paths = image_paths("images/")
cameras = pair_cameras(paths, read_cameras("transforms.json"))
views = Views(
    torch.stack([read_image(p) for p in paths]),  # all the same size
    torch.stack([c.k for c in cameras]),
    torch.stack([c.c2w for c in cameras]),
    normalized_intrinsics=True,
)
out = pipe(views)
```

## Layers

| | |
|---|---|
| `types` | `Views`, `Cameras`, `Gaussians` (+ `save_ply`, `read_ply_meta`) |
| `io` | camera files as `Camera`s (`read_cameras`, `save_cameras`, `pair_cameras`) and image helpers; independent of the rest |
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
(`preprocess`, returning the model input and its `ModelFrame`, or `None` without
input poses or when the model predicts its own cameras; `postprocess`, to model-frame `Gaussians`;
`render_planes`, near / far in world units, or in model units when the frame is
`None`; for models that predict cameras, `predicted_cameras`, from the model output
to model-frame `Cameras`, and `output_frame`, whose default fits the frame to the
given c2w as above; optionally `background` and `render_settings`, the model's
rasteriser settings such as `radius_clip`, `rasterize_mode` or `clamp`, which
`pipe.render` uses unless the caller passes them; `camera_input`, whether the
model is conditioned on the input cameras: `"required"` (the default, always),
`"optional"` (when given) or `"none"` (never);
for models trained on poses of a fixed scale, a `camera_spread` in the config, the
mean distance of the input cameras to the first one in the training data) and a
`ModelSpec(model_type, model_cls, processor_cls, display_name=...)`. Applications
read `camera_input`, `camera_spread` and `display_name` to handle a new
model without changes. `from_pretrained` resolves the `model_type` in
`config.json` in this order:

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

Document a new model in `docs/models/<family>.md` (source, licenses of code and
weights, zoo entries, usage notes, results against the official code; the existing
pages are the template) and add its row to
[`docs/models/index.md`](https://github.com/libgs/ffgs/blob/main/docs/models/index.md).

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

To release, set `version` in `pyproject.toml` (then `uv lock`), commit it as
`chore: release X.Y.Z`, merge it into main and push the tag `vX.Y.Z` on that commit.
`.github/workflows/release.yml` then runs CI, publishes to PyPI and creates the
[GitHub release](https://github.com/libgs/ffgs/releases), whose notes are generated
by [git-cliff](https://git-cliff.org/) from the conventional commits since the
previous tag (`cliff.toml`; preview with `uvx git-cliff --unreleased`).

## License

**The ffgs package as distributed is not entirely Apache-2.0: it includes code for
non-commercial use only.** Its license expression is
`Apache-2.0 AND MIT AND CC-BY-NC-4.0 AND CC-BY-NC-SA-4.0`:

- the core library is Apache-2.0 (see `LICENSE`); `src/ffgs/image.py` is adapted
  from pixelSplat (MIT), whose license is included at the end of `LICENSE`;
- vendored model code under `src/ffgs/models/<family>/` keeps its upstream license.
  `src/ffgs/models/anysplat/` mixes MIT, Apache-2.0 and non-commercial
  (CC-BY-NC-4.0, CC-BY-NC-SA-4.0) files; its `LICENSE` lists the license per file;
  `src/ffgs/models/c3g/` mixes MIT, Apache-2.0 and VGGT files kept with the
  CC-BY-NC-4.0 text; its `LICENSE` describes which is which;
- model weights are not part of the package and carry their own licenses.

The code and weight licenses of each model are listed in
[docs/models](https://github.com/libgs/ffgs/blob/main/docs/models/index.md).
