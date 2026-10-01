# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified by the libgs contributors, 2026, from
# src/model/encoder/vggt/layers/__init__.py of
# https://github.com/InternRobotics/AnySplat (5f5e208): `__all__` added (re-exports, not
# unused imports), imports sorted.
# "The license found in the LICENSE file" above is the one VGGT's code
# had when AnySplat vendored it (2025-06-30): CC BY-NC 4.0, non-commercial
# (facebookresearch/vggt c6bf698); see LICENSE (this directory).

from .attention import MemEffAttention
from .block import NestedTensorBlock
from .mlp import Mlp
from .patch_embed import PatchEmbed
from .swiglu_ffn import SwiGLUFFN, SwiGLUFFNFused

__all__ = [
    "MemEffAttention",
    "Mlp",
    "NestedTensorBlock",
    "PatchEmbed",
    "SwiGLUFFN",
    "SwiGLUFFNFused",
]
