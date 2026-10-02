"""TokenGS (Ren, Tyszkiewicz, Huang, Gojcic; CVPR 2026), vendored from the official
Apache-2.0 code at https://github.com/nv-tlabs/TokenGS (b16269c).

`model_type` "tokengs". Imported on first use: `import ffgs` does not import it.
"""

from __future__ import annotations

from typing import Any

from ...registry import ModelSpec, register_model
from .modeling import TokenGS
from .processing import TokenGSProcessor, TokenGSProcessorConfig

MODEL_TYPE = "tokengs"


def convert_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """An official checkpoint (the upstream module's state dict) -> `TokenGS`.

    The parameter names are the upstream ones; only the LPIPS training loss that
    upstream checkpoints may carry is dropped, as the upstream evaluation does.
    """
    return {k: v for k, v in checkpoint.items() if "lpips_loss" not in k}


SPEC = register_model(
    ModelSpec(
        model_type=MODEL_TYPE,
        model_cls=TokenGS,
        processor_cls=TokenGSProcessor,
        convert_state_dict=convert_state_dict,
    )
)

__all__ = [
    "MODEL_TYPE",
    "SPEC",
    "TokenGS",
    "TokenGSProcessor",
    "TokenGSProcessorConfig",
    "convert_state_dict",
]
