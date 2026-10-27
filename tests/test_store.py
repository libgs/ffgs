"""`ffgs.store`: records of the models ffgs added or used, over a fake HF cache."""

import huggingface_hub
import pytest
from _dummy import DummyProcessor
from _hub_cache import COMMIT, REPO, _cached, _ref
from test_pipeline import _model
from test_third_party import _fake_hub

from ffgs import GSPipeline, store, zoo


def test_home_follows_ffgs_home_then_xdg(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FFGS_HOME", str(tmp_path / "a"))
    assert store.home() == tmp_path / "a"
    monkeypatch.delenv("FFGS_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert store.home() == tmp_path / "xdg" / "ffgs"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert store.home() == tmp_path / "home" / ".cache" / "ffgs"


def test_sources_are_typed_like_from_pretrained(entries, tmp_path, monkeypatch) -> None:
    assert store.resolve("dummy/v1") == store.Record("dummy/v1", "zoo")
    assert store.resolve("someone/repo", revision="v2") == store.Record(
        "someone/repo", "hub", "v2"
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ckpt").mkdir()
    local = store.resolve("ckpt", subfolder="a")
    assert local == store.Record(str(tmp_path / "ckpt"), "local", None, "a")
    with pytest.raises(ValueError, match="pinned"):
        store.resolve("dummy/v1", revision="main")
    with pytest.raises(ValueError, match="local directory"):
        store.resolve("ckpt", revision="main")
    with pytest.raises(ValueError, match="not a directory, a zoo name"):
        store.resolve("no-such-thing")
    with pytest.raises(KeyError, match="dummy/v3"):
        store.resolve("dummy/v3")


def test_add_records_without_downloading(entries, cache, monkeypatch) -> None:
    def no_download(*args, **kwargs):
        raise AssertionError("add must not download")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", no_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_download)
    first = store.add("dummy/v1")
    assert first.added is not None and first.last_used is None
    assert store.add("dummy/v1") == first  # once per source
    store.add("someone/repo", revision="v2")
    assert [r.source for r in store.records()] == ["dummy/v1", "someone/repo"]
    assert store.status(first) == store.Status("not cached")
    assert store.get("someone/repo") is None  # another revision is another record
    assert store.get("someone/repo", revision="v2").type == "hub"


def test_zoo_status_follows_the_pinned_file(entries, cache, monkeypatch) -> None:
    record = store.add("dummy/v1")
    link = _cached(cache, REPO, COMMIT, "ckpts/v1.pt", entries["v1"])
    assert store.status(record) == store.Status("cached", len(entries["v1"]))
    link.unlink()  # deleted behind ffgs's back, e.g. `hf cache rm`
    assert store.status(record) == store.Status("not cached")
    (zoo._ROOT / "dummy" / "v1.json").unlink()  # gone from this version of ffgs
    assert store.status(record) == store.Status("unknown")


def test_hub_status_follows_the_snapshot(cache) -> None:
    record = store.add("someone/repo")
    assert store.status(record).state == "not cached"
    _cached(cache, "someone/repo", COMMIT, "config.json", b"{}")
    _cached(cache, "someone/repo", COMMIT, "model.safetensors", b"12345")
    _ref(cache, "someone/repo", "main", COMMIT)
    assert store.status(record) == store.Status("cached", 7)
    touched = store.touch("someone/repo")
    assert touched.commit == COMMIT and touched.last_used is not None

    variant = store.add("someone/repo", subfolder="re10k")
    assert store.status(variant).state == "not cached"
    _cached(cache, "someone/repo", COMMIT, "re10k/config.json", b"{}")
    _cached(cache, "someone/repo", COMMIT, "re10k/model.safetensors", b"123")
    assert store.status(variant) == store.Status("cached", 5)
    # The root pipeline leaves out a subfolder another record uses.
    assert store.status(touched) == store.Status("cached", 7)


def test_local_status(tmp_path) -> None:
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "ckpt" / "model.safetensors").write_bytes(b"1234")
    record = store.add(tmp_path / "ckpt")
    assert store.status(record) == store.Status("local", 4)
    (tmp_path / "ckpt" / "model.safetensors").unlink()
    (tmp_path / "ckpt").rmdir()
    assert store.status(record) == store.Status("missing")


def test_remove_deletes_records_never_files(entries, cache, tmp_path) -> None:
    link = _cached(cache, REPO, COMMIT, "ckpts/v1.pt", entries["v1"])
    store.add("dummy/v1")
    store.add("dummy/v2")
    assert [r.source for r in store.remove("dummy/v1")] == ["dummy/v1"]
    assert link.exists()  # the Hugging Face cache is cleaned with its own tools
    assert [r.source for r in store.records()] == ["dummy/v2"]
    with pytest.raises(KeyError, match="no record"):
        store.remove("dummy/v1")

    store.add("someone/repo")
    store.add("someone/repo", revision="v2", subfolder="a")
    assert len(store.remove("someone/repo")) == 2  # every revision and subfolder

    (tmp_path / "ckpt").mkdir()
    store.add(tmp_path / "ckpt")
    (tmp_path / "ckpt").rmdir()  # a deleted local directory is found by its path
    assert store.remove(tmp_path / "ckpt")[0].type == "local"
    assert [r.source for r in store.records()] == ["dummy/v2"]


def test_pull_checks_a_zoo_entry_and_records_a_hub_commit(
    entries, cache, tmp_path, monkeypatch
) -> None:
    upstream = tmp_path / "upstream"
    (upstream / "ckpts").mkdir(parents=True)
    (upstream / "ckpts" / "v1.pt").write_bytes(entries["v1"])
    (upstream / "ckpts" / "v2.pt").write_bytes(b"corrupt")
    calls = _fake_hub(monkeypatch, {REPO: upstream})
    assert store.pull("dummy/v1").type == "zoo"
    assert calls[-1]["revision"] == COMMIT
    with pytest.raises(ValueError, match="sha256"):
        store.pull("dummy/v2")

    def snapshot_download(repo_id, revision=None, allow_patterns=None):
        calls.append({"repo_id": repo_id, "allow_patterns": allow_patterns})
        link = _cached(cache, repo_id, COMMIT, "re10k/config.json", b"{}")
        return str(link.parent.parent)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    record = store.pull("someone/repo", subfolder="re10k")
    assert record.commit == COMMIT
    assert calls[-1]["allow_patterns"][0] == "re10k/config.json"
    with pytest.raises(FileNotFoundError, match="not a saved ffgs pipeline"):
        store.pull("someone/other")
    assert store.status(record).state == "cached"


def test_from_pretrained_records_what_it_loads(tmp_path, monkeypatch) -> None:
    saved = tmp_path / "saved"
    GSPipeline(_model(), DummyProcessor()).save_pretrained(saved)
    GSPipeline.from_pretrained(saved)
    (record,) = store.records()
    assert record.type == "local" and record.source == str(saved)
    assert record.last_used is not None

    _fake_hub(monkeypatch, {"someone/dummy": saved})
    GSPipeline.from_pretrained("someone/dummy", revision="v1")
    assert store.get("someone/dummy", revision="v1").last_used is not None
    GSPipeline.from_pretrained("someone/dummy", cache_dir=str(tmp_path / "own"))
    assert store.get("someone/dummy") is None  # the caller's own cache: not ours


def test_a_record_that_cannot_be_written_never_fails_a_load(
    tmp_path, monkeypatch
) -> None:
    saved = tmp_path / "saved"
    GSPipeline(_model(), DummyProcessor()).save_pretrained(saved)
    (tmp_path / "home").write_text("a file, not a directory")
    monkeypatch.setenv("FFGS_HOME", str(tmp_path / "home"))
    with pytest.warns(UserWarning, match="could not record"):
        GSPipeline.from_pretrained(saved)


def test_unreadable_records_are_skipped(cache) -> None:
    store.add("someone/repo")
    (store.home() / "models" / "broken.json").write_text("{not json")
    (store.home() / "models" / "odd.json").write_text('{"source": "x", "type": "?"}')
    assert [r.source for r in store.records()] == ["someone/repo"]
