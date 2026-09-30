import json
import sys
from pathlib import Path

import pytest
from safetensors import safe_open

FIXTURE = Path(__file__).parent / "fixtures" / "reference.safetensors"


def _has_gpu() -> bool:
    import importlib.util

    import torch

    return torch.cuda.is_available() and importlib.util.find_spec("gsplat") is not None


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--network", action="store_true", help="run tests marked `network`"
    )


def pytest_collection_modifyitems(config, items) -> None:
    skips = {}
    if not _has_gpu():
        skips["gpu"] = pytest.mark.skip(reason="needs CUDA and gsplat")
    if not config.getoption("--network"):
        skips["network"] = pytest.mark.skip(reason="needs --network")
    for item in items:
        for marker, skip in skips.items():
            if marker in item.keywords:
                item.add_marker(skip)


@pytest.fixture
def clean_registry():
    """Forget model types registered by a test, and the plugin / built-in model
    modules it imported (a built-in registers on import, so it must import again)."""
    from ffgs import registry

    before = dict(registry._REGISTRY)
    modules = set(sys.modules)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(before)
    for name in set(sys.modules) - modules:
        if name.startswith(("ffgs_test_plugin", "ffgs.models.")):
            del sys.modules[name]


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
