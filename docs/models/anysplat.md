# AnySplat

AnySplat ([paper](https://arxiv.org/abs/2505.23716),
[code](https://github.com/InternRobotics/AnySplat)) predicts Gaussians and the
input cameras from unposed images, on a VGGT-1B backbone. Poses are optional: given
c2w do not enter the model, they only fix the output frame (see
[Conventions](../../README.md#conventions)).

## Licenses

The code is vendored from the official code (`src/ffgs/models/anysplat/`, inference
only), each file under its own license: MIT (AnySplat), Apache-2.0 (DINOv2 layers),
CC BY-NC 4.0 (VGGT, as AnySplat vendored it) and CC BY-NC-SA 4.0 (Naver). **The
AnySplat model directory as a whole is for non-commercial use only**; see its
[`LICENSE`](../../src/ffgs/models/anysplat/LICENSE). The weights are MIT, but are
fine-tuned from VGGT-1B, whose weights are CC BY-NC 4.0 (non-commercial).

## Zoo entries

| zoo name | model | input views | input size | weights |
|---|---|---|---|---|
| `anysplat/default` | the released model: VGGT-1B backbone, SH degree 4 | any | 448×448 | [lhjiang/anysplat](https://huggingface.co/lhjiang/anysplat) |

## Usage

Without poses there is no world frame: the Gaussians and the predicted input
cameras are both in the model's frame, and renders take cameras in that frame:

```python
from ffgs import GSPipeline, Views

pipe = GSPipeline.from_pretrained("anysplat/default", device="cuda")
out = pipe(Views(images))                    # no intrinsics, no c2w: out.frame is None
out.gaussians.save_ply("scene.ply")          # in the model's frame
renders = pipe.render(out, out.cameras)["images"]  # the inputs, at the predicted cameras
```

A folder of images with
[`examples/infer_pose_free.py`](../../examples/infer_pose_free.py) (ply, predicted
cameras, re-rendered inputs):

```bash
python examples/infer_pose_free.py <image dir> --model anysplat/default \
    --num-views 8 --out out
```

The processor resizes with PIL bicubic by default, as upstream does
(`resize_mode="bicubic"`).

## Results on Mip-NeRF360

`anysplat/default` run with the authors' novel-view protocol (`src/eval_nvs.py`
of [InternRobotics/AnySplat](https://github.com/InternRobotics/AnySplat)
5f5e208: the images centre-cropped to 448×448, Gaussians and cameras from the
context images alone, target cameras from a second camera pass over context and
target images rescaled to the first) and with ffgs, on the same images and the
same weights. This is an agreement check with the official implementation, not a
reproduction of the paper's numbers (see the caveat below). Scene `room`, one view
set per setting:

| input views | official PSNR / SSIM / LPIPS | ffgs PSNR / SSIM / LPIPS | \|Δ\| PSNR / SSIM / LPIPS |
|---|---|---|---|
| 3 | 4.476 / 0.177 / 0.757 | 4.562 / 0.175 / 0.755 | 0.086 / 0.0018 / 0.0022 |
| 16 | 13.920 / 0.460 / 0.490 | 13.950 / 0.461 / 0.490 | 0.030 / 0.0010 / 0.0006 |

- **Inputs**: the preprocessed target images of both are identical.
- **Rendering**: the official Gaussians rendered by ffgs at the official cameras
  give the official metrics within 1e-4 (mean pixel |Δ| 1e-6, max 6e-3): ffgs's
  rotation + scale rasterisation of all views at once matches upstream's
  covariance, per-camera rendering.
- **Prediction**: not bit-identical on CUDA. After voxelisation the Gaussian counts
  differ by at most 0.07% (486 634 vs 486 305 at 3 views, 2 557 168 vs 2 557 438
  at 16 views) and the predicted context cameras by at most 5e-3; the metric
  differences above follow from that.
- **Caveat**: both are far below Table 1 of the paper (16.20 dB at 3 views, 21.85
  at 16). The authors did not publish their views; ours are spread evenly over
  the whole capture, which leaves much of each target unseen. At 3 views,
  upstream's target-camera scale (a ratio of signed mean translations) also came
  out negative (−0.47), which flips the target cameras. Both affect the official
  code and ffgs alike; we have not pursued them.
