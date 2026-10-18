# C3G

C3G ([paper](https://arxiv.org/abs/2512.04021),
[code](https://github.com/cvlab-kaist/C3G)) predicts a fixed budget of 2048
Gaussians from unposed images: learnable query tokens decode the features of a
VGGT aggregator. It predicts no cameras, and poses never enter the model: given
c2w only fix the output frame (see [Conventions](../../README.md#conventions)).

## Licenses

The code is vendored from the official code (`src/ffgs/models/c3g/`, Gaussian
inference only), each file under its own license: MIT (C3G), Apache-2.0 (DINOv2
layers) and the VGGT files with upstream's root-license pointer, kept with the
CC BY-NC 4.0 text. **The C3G model directory as a whole is for non-commercial
use only**; see its [`LICENSE`](../../src/ffgs/models/c3g/LICENSE). The authors
declare no license for the weights; they are fine-tuned from VGGT-1B, whose
weights are CC BY-NC 4.0 (non-commercial). The zoo downloads them from the
authors' repository; ffgs does not redistribute them.

## Zoo entries

| zoo name | model | input views | input size | weights |
|---|---|---|---|---|
| `c3g/default` | the released multiview Gaussian decoder: 2048 Gaussians, SH degree 0 | 2 or more | 224×224 | [honggyuAn/C3G](https://huggingface.co/honggyuAn/C3G) (`gaussian_decoder_multiview.ckpt`) |

The checkpoint is the authors' Lightning checkpoint (4.6 GB, with the frozen
DPT head it was trained with); the zoo pins its revision and SHA-256 and loads
the encoder strictly. The 2-view checkpoint and the feature decoders (LSeg,
DINOv2, DINOv3) are not in the zoo.

## Usage

Without poses there is no world frame and no cameras: the Gaussians are in the
model's frame (the first camera at the origin, looking down +z), and renders take
cameras in that frame:

```python
from ffgs import GSPipeline, Views

pipe = GSPipeline.from_pretrained("c3g/default", device="cuda")
out = pipe(Views(images))                    # out.frame and out.cameras are None
out.gaussians.save_ply("scene.ply")          # in the model's frame
```

With poses the output is placed in their frame, as upstream normalises the scene:
the first camera as the anchor and the distance between the first and last
cameras as the unit (it must lie in [1e-3, 1e10]):

```python
out = pipe(Views(images, intrinsics, c2w))   # c2w do not change the prediction
out.to_world().gaussians.save_ply("scene.ply")  # in the frame of c2w
renders = pipe.render(out, target_cameras)["images"]  # cameras in that frame
```

The processor resizes with PIL Lanczos and centre-crops to 224×224, as upstream
does. It renders as upstream: black background, no radius clip, no colour clamp.
Upstream's renderer is scale-invariant (it rescales near / far and the scene);
ffgs renders with gsplat at the nominal near / far 0.1 / 100, in the units of the
given c2w (of the model frame without poses).

## Results on RealEstate10K

`c3g/default` and the official code (cvlab-kaist/C3G 40b2cc9: the encoder and its
CUDA renderer, two context views, `test.align_pose=false`) on the same weights,
the first three scenes of the authors' RE10K evaluation index, each rendered at
its target cameras (normalised as upstream: first context camera, endpoint
baseline). This checks that ffgs runs the model as upstream does; it does not
reproduce the paper (no pose optimisation, three scenes).

| scene | official vs ffgs renders, PSNR | official vs GT, PSNR | ffgs vs GT, PSNR |
|---|---|---|---|
| `000c3ab189999a83` | 38.6 / 37.8 / 37.6 | 16.35 / 16.39 / 16.31 | 16.37 / 16.37 / 16.29 |
| `000db54a47bd43fe` | 37.0 / 35.5 / 35.2 | 21.45 / 20.76 / 20.24 | 21.49 / 20.83 / 20.31 |
| `0017ce4c6a39d122` | 40.0 / 40.2 / 40.0 | 19.20 / 16.82 / 16.61 | 19.15 / 16.81 / 16.60 |

- **Inputs**: the preprocessed context and target images are identical; the
  normalised target cameras agree to 1e-7.
- **Prediction**: on the CPU, the official encoder and ffgs give bit-identical
  aggregator tokens and Gaussian parameters. On CUDA neither is reproducible:
  run twice on the same input, the official model itself moves about 175 of the
  2048 Gaussians by more than 1e-2, as many as official vs ffgs. The cause is the
  head's attention: in fp32, `scaled_dot_product_attention` runs PyTorch's
  memory-efficient kernel (`fmha_cutlassF_f32_aligned_64x128_rf_sm80`, measured on
  an RTX PRO 6000 Blackwell with torch 2.7.1), whose output varies from run to run,
  also under `torch.use_deterministic_algorithms(True)`; the second decoder layer
  amplifies it. Every other op of the head is bit-reproducible, and so is the whole
  head with the math backend. ffgs keeps upstream's default backend.
- **Rendering**: the remaining render difference comes from that and from the
  renderers (upstream's scale-invariant CUDA rasteriser vs gsplat).
