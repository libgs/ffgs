"""AnySplat (Jiang, Mao, Xu, Lin, Xu, Gong, Dai, Wang, Lin, Zhou, Lin; ACM TOG /
SIGGRAPH Asia 2025), vendored from the official code at
https://github.com/InternRobotics/AnySplat (5f5e208).

Non-commercial: this directory holds MIT, CC BY-NC 4.0 (VGGT, as vendored by
AnySplat), CC BY-NC-SA 4.0 (Naver) and Apache-2.0 code, and the released weights
build on VGGT-1B; see LICENSE in this directory.

`model_type` "anysplat". Imported on first use: `import ffgs` does not import it.
"""

from __future__ import annotations

from typing import Any

from ...registry import ModelSpec, register_model
from .evaluation import predict_target_cameras, split_llffhold, target_cameras
from .modeling import AnySplat
from .processing import AnySplatProcessor, AnySplatProcessorConfig

MODEL_TYPE = "anysplat"


def convert_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """The official checkpoint (`lhjiang/anysplat` model.safetensors) -> `AnySplat`:
    the parameter names are the upstream ones, so it is used as it is."""
    return dict(checkpoint)


SPEC = register_model(
    ModelSpec(
        model_type=MODEL_TYPE,
        model_cls=AnySplat,
        processor_cls=AnySplatProcessor,
        convert_state_dict=convert_state_dict,
    )
)

__all__ = [
    "MODEL_TYPE",
    "SPEC",
    "AnySplat",
    "AnySplatProcessor",
    "AnySplatProcessorConfig",
    "convert_state_dict",
    "predict_target_cameras",
    "split_llffhold",
    "target_cameras",
]
