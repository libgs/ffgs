# Copyright (c) 2025 Lihan Jiang and Yucheng Mao
# SPDX-License-Identifier: MIT (see the AnySplat section of LICENSE, this directory)
#
# Modified by the libgs contributors, 2026, from src/eval_nvs.py of
# https://github.com/InternRobotics/AnySplat (5f5e208): the target-camera prediction
# of its `evaluate` as functions; the scale factor is taken per sample (upstream
# evaluates one scene at a time, where the two agree).

"""The official novel-view-synthesis protocol of AnySplat (`src/eval_nvs.py`).

AnySplat predicts its cameras; the target views of an evaluation are not posed in
its frame. Upstream places them there by running the aggregator and camera head
once more, on the context and target images together, and rescaling the predicted
target translations by the ratio of the mean context translations of the two
predictions:

>>> out = pipe(context)  # context views without c2w: out.frame is None
>>> cameras = target_cameras(pipe, out, context, targets)
>>> images = pipe.render(out, cameras)["images"]

Upstream holds out every `llffhold`-th image (`idx % 8 == 0`) as a target and uses
the others as context (`split_llffhold`).
"""

from __future__ import annotations

import torch

from ...types import Cameras, Views
from .modeling import AnySplat
from .vggt.utils.pose_enc import pose_encoding_to_extri_intri


def split_llffhold(num_views: int, llffhold: int = 8) -> tuple[list[int], list[int]]:
    """(context, target) view indices: targets are the indices divisible by
    `llffhold`, as `eval_nvs.py` splits a scene."""
    context = [i for i in range(num_views) if i % llffhold != 0]
    target = [i for i in range(num_views) if i % llffhold == 0]
    return context, target


def target_cameras(pipe, output, context: Views, targets: Views) -> Cameras:
    """Cameras of `targets` in the model frame of `output` (= `pipe(context)`, run
    without poses), by the official protocol. Render them with `pipe.render`."""
    if output.frame is not None or output.space != "model":
        raise ValueError(
            "the official protocol places targets in the model's own frame: run "
            "the pipeline on context views without c2w"
        )
    processor, device = pipe.processor, pipe.device
    context_images = processor.preprocess(context, device).model_input
    target_images = processor.preprocess(targets, device).model_input
    return predict_target_cameras(
        pipe.model, context_images, target_images, output.cameras.c2w
    )


@torch.no_grad()
def predict_target_cameras(
    model: AnySplat,
    context_images: torch.Tensor,
    target_images: torch.Tensor,
    context_c2w: torch.Tensor,
) -> Cameras:
    """`eval_nvs.py`'s target cameras: model inputs [B, V, 3, H, W] of the context
    and target views, and the context c2w [B, V, 4, 4] the encoder predicted for
    the context views alone (`pred_context_pose["extrinsic"]`)."""
    encoder = model.encoder
    b, num_context = context_images.shape[:2]
    images = torch.cat((context_images, target_images), dim=1).to(torch.bfloat16)
    h, w = images.shape[-2:]
    with torch.amp.autocast("cuda", enabled=False):
        tokens, _ = encoder.aggregator(
            images, intermediate_layer_idx=encoder.cfg.intermediate_layer_idx
        )
        tokens = [token.float() for token in tokens]
        pose_enc = encoder.camera_head(tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (h, w))

    padding = torch.tensor([0, 0, 0, 1], device=extrinsic.device, dtype=extrinsic.dtype)
    padding = padding.view(1, 1, 1, 4).repeat(b, images.shape[1], 1, 1)
    c2w = torch.cat([extrinsic, padding], dim=2).inverse()
    intrinsic = intrinsic.clone()
    intrinsic[:, :, 0] = intrinsic[:, :, 0] / w
    intrinsic[:, :, 1] = intrinsic[:, :, 1] / h

    # Upstream: the mean over all context translation components (signed).
    scale = context_c2w[:, :, :3, 3].flatten(1).mean(1) / c2w[
        :, :num_context, :3, 3
    ].flatten(1).mean(1)
    target_c2w = c2w[:, num_context:].clone()
    target_c2w[..., :3, 3] = target_c2w[..., :3, 3] * scale[:, None, None]
    return Cameras(
        target_c2w.float(),
        intrinsic[:, num_context:].float(),
        (h, w),
        normalized_intrinsics=True,
    )
