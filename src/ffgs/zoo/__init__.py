"""The model zoo: named, pinned pointers to weights in their authors' Hub repos.

An entry is one JSON file shipped in this package, `zoo/<family>/<variant>.json`,
loaded by name with `GSPipeline.from_pretrained("<family>/<variant>")`:

    {
      "model_type": "my-model",
      "description": "My model trained on X, 2 input views",
      "weights": {"repo": "author/repo", "file": "ckpt/model.pt",
                  "revision": "<40-hex commit>", "sha256": "<64-hex>"},
      "model": {...},          # model constructor arguments (its config.json)
      "processor": {...},      # its processor_config.json
      "source": {"url": "https://github.com/author/code", "paper": "https://..."},
      "license": "Apache-2.0"  # of the weights
    }

The weights stay where their authors put them; ffgs downloads the file at the
pinned commit, checks its sha256, and maps upstream key layouts to the model's
with the model's `ModelSpec.convert_state_dict`.

A zoo family name is reserved: `<family>/<anything>` names a zoo entry, never a Hub
repo (see `resolve`). Listing reads directory names only; an entry's JSON is read
when it is loaded.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import torch

from ..registry import ModelSpec

_ROOT = resources.files(__name__)
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REPO = re.compile(r"[\w.-]+/[\w.-]+")

_FIELDS = {
    # key: (type, required)
    "model_type": (str, True),
    "description": (str, True),
    "weights": (dict, True),
    "model": (dict, True),
    "processor": (dict, True),
    "source": (dict, True),
    "license": (str, True),
}
_WEIGHTS_FIELDS = {"repo": _REPO, "file": None, "revision": _COMMIT, "sha256": _SHA256}
_SOURCE_FIELDS = {"url": True, "paper": False}


@dataclass(frozen=True)
class Weights:
    repo: str
    file: str
    revision: str  # a full commit sha, never a branch or tag
    sha256: str


@dataclass(frozen=True)
class ZooEntry:
    name: str  # "<family>/<variant>"
    model_type: str
    description: str
    weights: Weights
    model: dict[str, Any]
    processor: dict[str, Any]
    source: dict[str, str]
    license: str


def families() -> list[str]:
    return sorted(
        d.name for d in _ROOT.iterdir() if d.is_dir() and _NAME.fullmatch(d.name)
    )


def list_models(family: str | None = None) -> list[str]:
    """Names of the shipped entries (`"<family>/<variant>"`); reads no entry."""
    names = []
    for fam in families() if family is None else [family]:
        directory = _ROOT / fam
        if not directory.is_dir():
            continue
        for file in directory.iterdir():
            variant = file.name.removesuffix(".json")
            if file.name.endswith(".json") and _NAME.fullmatch(variant):
                names.append(f"{fam}/{variant}")
    return sorted(names)


def resolve(name: str | Path) -> str | None:
    """The zoo entry `name` refers to, or None if it is not a zoo name.

    `name` is a zoo name when it is a string `"<family>/<variant>"` whose family
    ships with ffgs and that is not an existing local directory. Such a name never
    falls through to the Hub: an unknown variant is an error listing the known ones.
    """
    if not isinstance(name, str) or Path(name).is_dir():
        return None
    family, sep, variant = name.partition("/")
    if not sep or not _NAME.fullmatch(family) or not (_ROOT / family).is_dir():
        return None
    if (
        not _NAME.fullmatch(variant)
        or not (_ROOT / family / f"{variant}.json").is_file()
    ):
        raise KeyError(
            f"no zoo entry {name!r}; {family!r} has {list_models(family)}. "
            f"Zoo families are reserved names: to load a Hub repo of an owner "
            f"called {family!r}, download it (huggingface_hub.snapshot_download) "
            "and pass the directory."
        )
    return name


def get_entry(name: str) -> ZooEntry:
    """Read and validate the entry `name` (`"<family>/<variant>"`)."""
    family, _, variant = name.partition("/")
    file = _ROOT / family / f"{variant}.json"
    if not (_NAME.fullmatch(family) and _NAME.fullmatch(variant) and file.is_file()):
        raise KeyError(f"no zoo entry {name!r}; known: {list_models()}")
    try:
        data = json.loads(file.read_text())
    except json.JSONDecodeError as err:
        raise ValueError(f"zoo entry {name}: invalid JSON ({err})") from None
    return parse_entry(name, data)


def parse_entry(name: str, data: Any) -> ZooEntry:
    """Validate an entry's JSON against the schema in the module docstring."""

    def fail(message: str) -> ValueError:
        return ValueError(f"zoo entry {name}: {message}")

    if not isinstance(data, dict):
        raise fail("must be a JSON object")
    _check_keys(data, set(_FIELDS), {k for k, (_, r) in _FIELDS.items() if r}, fail)
    for key, (kind, _) in _FIELDS.items():
        if not isinstance(data[key], kind):
            raise fail(
                f"{key} must be {'a JSON object' if kind is dict else 'a string'}"
            )

    weights = data["weights"]
    _check_keys(weights, set(_WEIGHTS_FIELDS), set(_WEIGHTS_FIELDS), fail, "weights.")
    for key, pattern in _WEIGHTS_FIELDS.items():
        value = weights[key]
        if not isinstance(value, str) or not value:
            raise fail(f"weights.{key} must be a non-empty string")
        if pattern is not None and not pattern.fullmatch(value):
            hint = {
                "repo": "an 'owner/name' Hub repo id",
                "revision": "a full 40-hex commit sha (not a branch or tag)",
                "sha256": "64 lowercase hex digits",
            }[key]
            raise fail(f"weights.{key} = {value!r} must be {hint}")

    source = data["source"]
    required = {k for k, r in _SOURCE_FIELDS.items() if r}
    _check_keys(source, set(_SOURCE_FIELDS), required, fail, "source.")
    if not all(isinstance(v, str) and v for v in source.values()):
        raise fail("source values must be non-empty strings")

    return ZooEntry(
        name=name,
        model_type=data["model_type"],
        description=data["description"],
        weights=Weights(**weights),
        model=data["model"],
        processor=data["processor"],
        source=dict(source),
        license=data["license"],
    )


def load_state_dict(
    entry: ZooEntry, spec: ModelSpec, **hub_kwargs: Any
) -> dict[str, torch.Tensor]:
    """Download the entry's weights at the pinned commit, check the sha256, and
    convert them to `spec.model_cls`'s layout (`spec.convert_state_dict`)."""
    from huggingface_hub import hf_hub_download

    w = entry.weights
    path = Path(
        hf_hub_download(
            repo_id=w.repo, filename=w.file, revision=w.revision, **hub_kwargs
        )
    )
    digest = sha256_file(path)
    if digest != w.sha256:
        raise ValueError(
            f"zoo entry {entry.name}: sha256 of {w.repo}/{w.file}@{w.revision} is "
            f"{digest}, expected {w.sha256} ({path}). The download may be corrupt "
            "(retry with force_download=True) or the entry is wrong."
        )
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        checkpoint: Any = load_file(path)
    else:
        # Plain tensors only: a checkpoint that needs arbitrary unpickling is
        # not loaded.
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if spec.convert_state_dict is not None:
        return spec.convert_state_dict(checkpoint)
    return checkpoint


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_keys(data, allowed, required, fail, prefix="") -> None:
    if not isinstance(data, dict):
        raise fail(f"{prefix.rstrip('.')} must be a JSON object")
    unknown = set(data) - allowed
    if unknown:
        raise fail(f"unknown keys {sorted(prefix + k for k in unknown)}")
    missing = required - set(data)
    if missing:
        raise fail(f"missing keys {sorted(prefix + k for k in missing)}")
