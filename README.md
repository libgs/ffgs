# ffgs — standard inference for feed-forward 3D Gaussian Splatting

One way to load and run feed-forward 3D Gaussian Splatting models: posed images in,
Gaussians out — in the model's own frame, or in **your** world frame with one call —
whatever model is behind it.

> Early development: the core interface, third-party loading and the model zoo
> are in place; models are being added.

```python
from ffgs import Cameras, GSPipeline, Views

pipe = GSPipeline.from_pretrained("family/variant", device="cuda")  # or a path / Hub repo
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

## Conventions

| | |
|---|---|
| images | float in [0, 1] (uint8 accepted), `[V, 3, H, W]` or `[B, V, 3, H, W]`, any resolution |
| c2w | camera-to-world, OpenCV axes (x right, y down, z forward) |
| intrinsics | pixel K for the image size given, or normalised K with `normalized_intrinsics=True` |
| output | `GSOutput`: `Gaussians` (means, scales, wxyz quats, opacities, SH or RGB colours) in the model's frame, and that frame |

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
(`preprocess`, returning the model input and its `ModelFrame`; `postprocess`, to
model-frame `Gaussians`; `render_planes`, near / far in world units) and a
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
