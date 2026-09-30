# ffgs — standard inference for feed-forward 3D Gaussian Splatting

One way to load and run feed-forward 3D Gaussian Splatting models: posed images in,
Gaussians in **your** world frame out, whatever model is behind it.

> Early development: the core interface is in place; models, the model zoo and
> third-party loading are being added.

```python
from ffgs import Cameras, GSPipeline, Views

pipe = GSPipeline.from_pretrained("path/or/hf-repo-id", device="cuda")
out = pipe(Views(images, intrinsics, c2w))          # -> GSOutput
out.gaussians.save_ply("scene.ply")                  # standard 3DGS ply
images = pipe.render(out, Cameras(c2w_new, k_new, (h, w)))["images"]
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
| output | `Gaussians` (means, scales, wxyz quats, opacities, SH or RGB colours) in the world frame of the input c2w |

A processor reproduces its model's training preprocessing: rescale to cover the
model's input size, centre crop (normalised principal point kept, as in training),
cameras relative to the first input view, translations times a model-specific
`scene_scale`. Predictions are mapped back by the inverse similarity, SH included.

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

A saved model is a directory (or Hub repo) with `config.json` (model config +
`model_type`), `model.safetensors` and `processor_config.json`;
`GSPipeline.save_pretrained` writes all three.

Adding a model: a model class with `PyTorchModelHubMixin`, a `Processor` subclass
(`preprocess`, `postprocess`, `render_planes`), and `register_model(ModelSpec(...))`.

## Development

```bash
uv sync
uv run pytest            # everything; tests marked `gpu` skip without CUDA + gsplat
uv run pytest -m gpu     # only the GPU tests (needs CUDA and `--extra render`)
uv run pre-commit install
```

`uv sync` installs torch from PyPI (CUDA build). CI has no GPU and installs CPU
torch first (see `.github/workflows/ci.yml`). `tests/fixtures/reference.safetensors`
holds frozen reference outputs of the default data path; change it only together
with a deliberate change to that path.

## License

Apache-2.0 (see `LICENSE`). `src/ffgs/image.py` is adapted from pixelSplat (MIT);
its license is included at the end of `LICENSE`.
