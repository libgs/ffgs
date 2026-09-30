"""Model code shipped inside a model repo, named by `config.json:auto_map`.

The convention of transformers / diffusers: a repo whose model is not installed
lists its own classes,

    "auto_map": {"model": "modeling_foo.FooModel",
                 "processor": "processing_foo.FooProcessor"}

and the `.py` files next to `config.json` are imported only when the caller passes
`trust_remote_code=True`. The files are imported as one package per directory
(Hub snapshot or local directory), so relative imports between them work.
`export_code` writes such a directory from an installed package
(`GSPipeline.save_pretrained(..., include_code=True)`).
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.machinery
import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from .processor import Processor
from .registry import ModelSpec

AUTO_MAP_KEYS = ("model", "processor")

# Parent package of every imported code directory.
_PACKAGE = "ffgs_remote_code"


def parse_auto_map(auto_map: Any) -> dict[str, tuple[str, str]]:
    """`{"model": "module.Class", ...}` -> `{"model": ("module", "Class"), ...}`."""
    if not isinstance(auto_map, dict) or set(auto_map) != set(AUTO_MAP_KEYS):
        raise ValueError(
            f"config.json:auto_map must map exactly {list(AUTO_MAP_KEYS)} to "
            f'"module.Class" references, got {auto_map!r}'
        )
    parsed = {}
    for key, reference in auto_map.items():
        if not isinstance(reference, str) or "." not in reference:
            raise ValueError(
                f'auto_map["{key}"] must be "module.Class", got {reference!r}'
            )
        if "--" in reference:
            raise ValueError(
                f'auto_map["{key}"] = {reference!r} points at another repo; '
                "ffgs only runs code from the repo being loaded, copy it there"
            )
        module, name = reference.rsplit(".", 1)
        parsed[key] = (module, name)
    return parsed


def spec_from_auto_map(
    model_type: str, auto_map: Any, pretrained: str | Path, **hub_kwargs: Any
) -> ModelSpec:
    """Import the classes `auto_map` names from `pretrained` (runs its code)."""
    parsed = parse_auto_map(auto_map)
    package = _import_package(_code_directory(pretrained, **hub_kwargs))
    classes = {
        key: _load_class(package, module, name, pretrained)
        for key, (module, name) in parsed.items()
    }
    if not issubclass(classes["processor"], Processor):
        raise TypeError(
            f"auto_map processor {classes['processor'].__name__} of {pretrained} "
            "does not subclass ffgs.Processor"
        )
    return ModelSpec(model_type, classes["model"], classes["processor"])


def save_code(
    model_cls: type, processor_cls: type, save_directory: Path
) -> dict[str, str] | None:
    """Copy the repo code `model_cls` / `processor_cls` came from (every `.py`
    file of its directory) into `save_directory` and return the matching
    auto_map; None for installed code."""
    remote = [_is_remote(cls) for cls in (model_cls, processor_cls)]
    if not any(remote):
        return None
    if not all(remote) or model_cls.__module__.split(".")[1] != (
        processor_cls.__module__.split(".")[1]
    ):
        raise ValueError(
            "cannot save a model and processor that come from different code "
            "sources as one repo"
        )
    package = ".".join(model_cls.__module__.split(".")[:2])
    source = Path(sys.modules[package].__path__[0])
    # All of them, as they are downloaded: a module the model imports only when
    # it runs is not imported yet.
    for file in sorted(p for p in source.rglob("*.py") if "__pycache__" not in p.parts):
        target = save_directory / file.relative_to(source)
        if target.exists() and target.samefile(file):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    return {
        key: f"{cls.__module__[len(package) + 1:]}.{cls.__qualname__}"
        for key, cls in zip(AUTO_MAP_KEYS, (model_cls, processor_cls))
    }


def export_code(
    model_cls: type, processor_cls: type, save_directory: Path
) -> dict[str, str]:
    """Copy the installed package holding `model_cls` and `processor_cls` (every
    `.py` file under it) into `save_directory` and return the matching auto_map,
    so the directory loads with `trust_remote_code=True` where that package is not
    installed.

    The package is the innermost one holding both classes. Its code may import
    ffgs and third-party packages, and itself only relatively: an absolute import
    of its own distribution, or a relative one reaching above the package, would
    not resolve in the copy and is refused.
    """
    package = _common_package(model_cls, processor_cls)
    top = package.split(".")[0]
    if top == "ffgs":
        raise ValueError(
            f"{model_cls.__name__} is built into ffgs; its directories load "
            "without shipping code"
        )
    source = Path(sys.modules[package].__path__[0])
    files = sorted(p for p in source.rglob("*.py") if "__pycache__" not in p.parts)
    problems = [
        problem for file in files for problem in _outside_imports(file, source, top)
    ]
    if problems:
        raise ValueError(
            f"cannot ship package {package!r} as model code: it imports code "
            "that is not in it (import within the package relatively):\n  "
            + "\n  ".join(problems)
        )
    for file in files:
        target = save_directory / file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, target)
    return {
        key: f"{cls.__module__[len(package) + 1:]}.{cls.__qualname__}"
        for key, cls in zip(AUTO_MAP_KEYS, (model_cls, processor_cls))
    }


def _common_package(*classes: type) -> str:
    paths = []
    for cls in classes:
        module = sys.modules[cls.__module__]
        name = module.__name__
        paths.append(name if hasattr(module, "__path__") else name.rpartition(".")[0])
    common = []
    if all(paths):
        for names in zip(*(path.split(".") for path in paths)):
            if len(set(names)) > 1:
                break
            common.append(names[0])
    if not common:
        raise ValueError(
            "the model and processor classes must live in one package to be "
            f"shipped as model code, got {[cls.__module__ for cls in classes]}"
        )
    return ".".join(common)


def _outside_imports(file: Path, source: Path, top: str) -> list[str]:
    """Imports in `file` that would not resolve once `source` is copied alone."""
    depth = len(file.parent.relative_to(source).parts)
    problems = []
    for node in ast.walk(ast.parse(file.read_text(), filename=str(file))):
        if isinstance(node, ast.ImportFrom) and node.level:
            if node.level - 1 > depth:
                where = f"{file.relative_to(source)}:{node.lineno}"
                problems.append(f"{where}: {'.' * node.level}{node.module or ''}")
            continue
        if isinstance(node, ast.ImportFrom):
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        for name in names:
            if name.split(".")[0] == top:
                problems.append(f"{file.relative_to(source)}:{node.lineno}: {name}")
    return problems


def _is_remote(cls: type) -> bool:
    return cls.__module__.startswith(_PACKAGE + ".")


def _code_directory(pretrained: str | Path, **hub_kwargs: Any) -> Path:
    local = Path(pretrained)
    if local.is_dir():
        return local.resolve()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(str(pretrained), allow_patterns=["*.py"], **hub_kwargs)
    )


def _import_package(directory: Path) -> str:
    """Register `directory` as a package (its `__init__.py` is not run)."""
    if _PACKAGE not in sys.modules:
        root = importlib.machinery.ModuleSpec(_PACKAGE, None, is_package=True)
        root.submodule_search_locations = []
        sys.modules[_PACKAGE] = importlib.util.module_from_spec(root)
    digest = hashlib.sha256(str(directory).encode()).hexdigest()[:16]
    name = f"{_PACKAGE}.m{digest}"
    if name not in sys.modules:
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        spec.submodule_search_locations = [str(directory)]
        module: ModuleType = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        setattr(sys.modules[_PACKAGE], f"m{digest}", module)
    return name


def _load_class(package: str, module: str, name: str, pretrained: str | Path) -> type:
    try:
        imported = importlib.import_module(f"{package}.{module}")
    except ModuleNotFoundError as err:
        if err.name != f"{package}.{module}":
            raise
        raise ModuleNotFoundError(
            f"auto_map names module {module!r} but {module.replace('.', '/')}.py "
            f"is not in {pretrained}"
        ) from None
    cls = getattr(imported, name, None)
    if not isinstance(cls, type):
        raise ImportError(f"{module}.py of {pretrained} defines no class {name!r}")
    return cls
