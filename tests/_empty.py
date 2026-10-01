"""Build large models without allocating their weights."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from torch import nn


@contextmanager
def meta_parameters() -> Iterator[None]:
    """Parameters created inside go to the meta device as they are registered.

    For models whose construction calls `.item()` (which `torch.device("meta")`
    cannot run); buffers stay where they are created.
    """
    register = nn.Module.register_parameter

    def to_meta(module: nn.Module, name: str, param: nn.Parameter | None) -> None:
        register(module, name, param)
        if param is not None:
            module._parameters[name] = nn.Parameter(
                param.to("meta"), requires_grad=param.requires_grad
            )

    nn.Module.register_parameter = to_meta
    try:
        yield
    finally:
        nn.Module.register_parameter = register
