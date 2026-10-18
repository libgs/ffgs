"""C3G @ 40b2cc9, inference only; see LICENSE for non-commercial restrictions.

The original Lightning checkpoint is loaded with weights_only=True by the zoo
loader. Only its state_dict is used, stripping exactly one encoder. prefix;
all 1300 inference parameters (including visualization DPT) load strictly.
Neither Lightning nor a separate VGGT checkpoint is needed.
"""

from __future__ import annotations

from typing import Any

import torch

from ...registry import ModelSpec, register_model
from .modeling import C3G, C3GOutput
from .processing import C3GProcessor, C3GProcessorConfig

MODEL_TYPE = "c3g"


def convert_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("C3G expects the official Lightning checkpoint state_dict")
    if any(not name.startswith("encoder.") for name in state):
        raise ValueError("C3G state_dict contains unexpected non-encoder keys")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("C3G state_dict must contain tensors only")
    return {name.removeprefix("encoder."): value for name, value in state.items()}


SPEC = register_model(
    ModelSpec(
        model_type=MODEL_TYPE,
        model_cls=C3G,
        processor_cls=C3GProcessor,
        convert_state_dict=convert_state_dict,
        display_name="C3G",
    )
)

__all__ = [
    "C3G",
    "C3GOutput",
    "C3GProcessor",
    "C3GProcessorConfig",
    "MODEL_TYPE",
    "SPEC",
    "convert_state_dict",
]
