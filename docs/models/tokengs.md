# TokenGS

TokenGS ([paper](https://arxiv.org/abs/2604.15239),
[code](https://github.com/nv-tlabs/TokenGS)) predicts Gaussians from posed images
through a fixed set of Gaussian tokens. It is trained on DL3DV and needs intrinsics
and c2w for every input view.

## Licenses

The code is vendored under Apache-2.0 (`src/ffgs/models/tokengs/`, feed-forward
inference only: no test-time training). **The weights are under the NVIDIA Internal
Scientific Research and Development Model License (non-commercial).**

## Zoo entries

| zoo name | model | input views | input size | weights |
|---|---|---|---|---|
| `tokengs/dl3dv-base` | 1024 Gaussian tokens (the base the variants below are finetuned from) | 4 | 256×256 | [jiaweir/tokengs](https://huggingface.co/jiaweir/tokengs) |
| `tokengs/dl3dv-{2,4,6}v` | 4096 Gaussian tokens | 2 / 4 / 6 | 256×448 | [jiaweir/tokengs](https://huggingface.co/jiaweir/tokengs) |
| `tokengs/dl3dv-latent-{2,4,6}v-{ssim,lpips}` | 2026.6 latent-bottleneck models, trained with an SSIM or an LPIPS loss | 2 / 4 / 6 | 256×448 | [jiaweir/tokengs](https://huggingface.co/jiaweir/tokengs) |

## Usage

```python
from ffgs import GSPipeline, Views

pipe = GSPipeline.from_pretrained("tokengs/dl3dv-6v", device="cuda")
out = pipe(Views(images, intrinsics, c2w))      # poses are required
out.to_world().gaussians.save_ply("scene.ply")  # in the frame of c2w
```

A DL3DV scene (nerfstudio layout) with
[`examples/infer_nerfstudio_scene.py`](../../examples/infer_nerfstudio_scene.py):

```bash
python examples/infer_nerfstudio_scene.py <scene> --model tokengs/dl3dv-6v \
    --images-dir <scene>/images_4 --out out
```

## Results on DL3DV

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
- The unmodified upstream `evaluate.py` matches our evaluation of the official
  code within 5e-5 on the paper and finetuned rows. It cannot run the
  `latent-*-lpips` presets in this setup: it also computes the training loss,
  whose LPIPS term fails under bf16 autocast, so our evaluation disables that
  (discarded) loss term.
