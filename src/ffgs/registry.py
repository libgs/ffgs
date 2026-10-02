"""model_type -> (model class, processor class).

A `model_type` is looked up in order:

1. models registered in this process (`register_model`), including built-ins,
   which are imported only when their `model_type` is first asked for;
2. installed plugins: the `ffgs.models` entry point group, entry point name =
   model_type, value = a `ModelSpec` (or a module that calls `register_model`
   on import). Plugins are loaded only when their `model_type` is asked for;
3. code shipped in the model repo (`config.json:auto_map`), only with
   `trust_remote_code=True` — see `ffgs.dynamic` and `GSPipeline.from_pretrained`.

`import ffgs` imports no plugin and runs no repo code.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from .processor import Processor

ENTRY_POINT_GROUP = "ffgs.models"

# model_type -> module that calls `register_model` for it on import.
_BUILTIN: dict[str, str] = {"tokengs": "ffgs.models.tokengs"}


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    model_cls: type
    processor_cls: type[Processor]
    # Upstream checkpoint (as loaded from a zoo entry's weights file) -> state dict
    # of `model_cls`. Not applied to directories written by `save_pretrained`.
    convert_state_dict: Callable[[Any], dict[str, Any]] | None = None


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
        plugins = _plugin_entry_points().get(model_type, [])
        if len(plugins) > 1:
            raise ValueError(
                f"model_type {model_type!r} is provided by several installed "
                f"plugins: {[ep.value for ep in plugins]}; uninstall all but one"
            )
        if plugins:
            _load_plugin(plugins[0])
    if model_type not in _REGISTRY:
        raise KeyError(
            f"unknown model_type {model_type!r}; known: {known_model_types()}. "
            "Install the package that provides it (found through the "
            f"{ENTRY_POINT_GROUP!r} entry point group) or call "
            "ffgs.register_model(...) before loading. A model repo can also ship "
            "its own code under config.json:auto_map, loaded with "
            "GSPipeline.from_pretrained(..., trust_remote_code=True)."
        )
    return _REGISTRY[model_type]


def known_model_types() -> list[str]:
    """Registered, built-in and installed-plugin model types (nothing imported)."""
    return sorted(set(_REGISTRY) | set(_BUILTIN) | set(_plugin_entry_points()))


def spec_for_model(model: object) -> ModelSpec:
    for model_type in _BUILTIN:
        get_model_spec(model_type)
    for model_type, plugins in _plugin_entry_points().items():
        if model_type in _REGISTRY:
            continue
        try:
            get_model_spec(model_type)
        except Exception as err:  # a broken plugin must not hide the others
            warnings.warn(
                f"skipping ffgs plugin {[ep.value for ep in plugins]}: {err}",
                stacklevel=2,
            )
    # The model's own class first: a plugin may subclass another model.
    for spec in _REGISTRY.values():
        if type(model) is spec.model_cls:
            return spec
    for spec in _REGISTRY.values():
        if isinstance(model, spec.model_cls):
            return spec
    raise KeyError(f"no model_type registered for {type(model).__name__}")


def _plugin_entry_points() -> dict[str, list[EntryPoint]]:
    found: dict[str, list[EntryPoint]] = {}
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        # The same distribution can be visible twice on sys.path.
        if all(other.value != ep.value for other in found.get(ep.name, [])):
            found.setdefault(ep.name, []).append(ep)
    return found


def _load_plugin(ep: EntryPoint) -> None:
    loaded = ep.load()
    if isinstance(loaded, ModelSpec):
        if loaded.model_type != ep.name:
            raise ValueError(
                f"entry point {ep.name!r} = {ep.value!r} is a ModelSpec for "
                f"model_type {loaded.model_type!r}; the entry point name must "
                "be the model_type"
            )
        register_model(loaded)
    elif ep.name not in _REGISTRY:
        raise TypeError(
            f"entry point {ep.name!r} = {ep.value!r} must be a ModelSpec, or a "
            f"module that calls register_model for {ep.name!r} on import"
        )
