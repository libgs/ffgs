import json
from pathlib import Path

import pytest
from safetensors import safe_open

FIXTURE = Path(__file__).parent / "fixtures" / "reference.safetensors"


def _has_gpu() -> bool:
    import importlib.util

    import torch

    return torch.cuda.is_available() and importlib.util.find_spec("gsplat") is not None


def pytest_collection_modifyitems(config, items) -> None:
    if _has_gpu():
        return
    skip = pytest.mark.skip(reason="needs CUDA and gsplat")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


class Reference:
    """Frozen reference outputs (tensors + metadata) of the default data path."""

    def __init__(self, path: Path) -> None:
        with safe_open(str(path), framework="pt") as handle:
            self.metadata = handle.metadata()
            self.tensors = {k: handle.get_tensor(k) for k in handle.keys()}

    def meta(self, key: str):
        return json.loads(self.metadata[key])

    def __getitem__(self, key: str):
        return self.tensors[key]


@pytest.fixture(scope="session")
def reference() -> Reference:
    return Reference(FIXTURE)
