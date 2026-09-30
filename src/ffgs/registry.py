"""model_type -> (model class, processor class).

Built-in models are imported only when their `model_type` is first asked for, so
`import ffgs` stays light. Third-party models call `register_model`.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

from .processor import Processor

# model_type -> module that calls `register_model` for it on import.
_BUILTIN: dict[str, str] = {}


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    model_cls: type
    processor_cls: type[Processor]


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> ModelSpec:
    existing = _REGISTRY.get(spec.model_type)
    if existing is not None and existing != spec:
        raise ValueError(f"model_type {spec.model_type!r} is already registered")
    _REGISTRY[spec.model_type] = spec
    return spec


def get_model_spec(model_type: str) -> ModelSpec:
    if model_type not in _REGISTRY and model_type in _BUILTIN:
        importlib.import_module(_BUILTIN[model_type])
    if model_type not in _REGISTRY:
        known = sorted(set(_REGISTRY) | set(_BUILTIN))
        raise KeyError(f"unknown model_type {model_type!r}; known: {known}")
    return _REGISTRY[model_type]


def spec_for_model(model: object) -> ModelSpec:
    for model_type in _BUILTIN:
        get_model_spec(model_type)
    for spec in _REGISTRY.values():
        if isinstance(model, spec.model_cls):
            return spec
    raise KeyError(f"no model_type registered for {type(model).__name__}")
