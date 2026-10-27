"""Fixtures of a fake Hugging Face cache and zoo (a pytest plugin, see conftest)."""

import hashlib
import json
import os

import huggingface_hub.constants
import pytest

from ffgs import zoo

REPO = "author/upstream"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """An empty Hugging Face cache (`HF_HUB_CACHE`)."""
    root = tmp_path / "hf-cache"
    root.mkdir()
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(root))
    return root


def _cached(cache, repo, commit, name, data: bytes):
    """A file of a snapshot in the cache layout: a link to a content blob."""
    repo_dir = cache / ("models--" + repo.replace("/", "--"))
    blob = repo_dir / "blobs" / hashlib.sha256(data).hexdigest()
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(data)
    link = repo_dir / "snapshots" / commit / name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(blob, link.parent))
    return link


def _ref(cache, repo, ref, commit) -> None:
    path = cache / ("models--" + repo.replace("/", "--")) / "refs" / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(commit)


@pytest.fixture
def entries(tmp_path, monkeypatch):
    """A zoo family `dummy` whose two entries are files of one repo revision."""
    root = tmp_path / "zoo"
    (root / "dummy").mkdir(parents=True)
    monkeypatch.setattr(zoo, "_ROOT", root)
    weights = {"v1": b"weights one", "v2": b"weights two"}
    for variant, data in weights.items():
        entry = {
            "model_type": "ffgs-test-dummy",
            "description": f"dummy {variant}",
            "weights": {
                "repo": REPO,
                "file": f"ckpts/{variant}.pt",
                "revision": COMMIT,
                "sha256": hashlib.sha256(data).hexdigest(),
            },
            "model": {},
            "processor": {},
            "source": {"url": "https://example.com"},
            "license": "MIT",
        }
        (root / "dummy" / f"{variant}.json").write_text(json.dumps(entry))
    return weights
