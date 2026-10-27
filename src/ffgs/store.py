"""ffgs's records of its models: what was added or used, and where its weights are.

The weights (and code) stay in the Hugging Face cache, shared with other tools;
ffgs keeps one JSON record per model under `$FFGS_HOME/models` (by default
`$XDG_CACHE_HOME/ffgs`, else `~/.cache/ffgs`):

    {"source": "someone/repo", "type": "hub", "revision": null,
     "subfolder": null, "commit": "<40 hex>", "added": "...", "last_used": "..."}

`type` is `"zoo"` (a zoo name), `"hub"` (a Hub repo id) or `"local"` (a directory,
stored as its absolute path). A record holds no file list: the files of a zoo
entry are its pinned weights file; those of a Hub model are its snapshot in the
cache (`snapshots/<commit>[/<subfolder>]`); a local directory is its own. Whether
they are there is checked when asked (`status`), so a record never claims files
that were deleted behind its back: they are `"not cached"`, and downloaded again
when the model is next used.

`GSPipeline.from_pretrained` records every model it loads (unless given its own
`cache_dir`); `add` records one without downloading it, `pull` downloads it and
`remove` deletes its records — never files: the Hugging Face cache is cleaned with
its own tools, a local directory is the user's.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

from . import zoo

TYPES = ("zoo", "hub", "local")
# `status` states: the weights are in the Hugging Face cache, or not (never
# downloaded, or deleted since); the local directory is there, or not; the zoo
# entry is gone from this version of ffgs.
STATES = ("cached", "not cached", "local", "missing", "unknown")

_COMMIT = re.compile(r"[0-9a-f]{40}")
_REPO = re.compile(r"[\w.-]+/[\w.-]+")
# What a saved pipeline is made of (`GSPipeline.save_pretrained`).
PIPELINE_FILES = (
    "config.json",
    "model.safetensors",
    "pytorch_model.bin",
    "processor_config.json",
    "*.py",
)


@dataclass(frozen=True)
class Record:
    source: str  # zoo name, Hub repo id or absolute directory path
    type: str  # one of TYPES
    revision: str | None = None  # as asked for (None: the default branch)
    subfolder: str | None = None
    commit: str | None = None  # of the last download, for the Hub
    added: str | None = None
    last_used: str | None = None

    @property
    def key(self) -> tuple[str, str | None, str | None]:
        return (self.source, self.revision, self.subfolder)


@dataclass(frozen=True)
class Status:
    state: str  # one of STATES
    size: int = 0  # bytes on disk (0 unless cached or local)


def home() -> Path:
    """`$FFGS_HOME`, else `$XDG_CACHE_HOME/ffgs`, else `~/.cache/ffgs`."""
    if os.environ.get("FFGS_HOME"):
        return Path(os.environ["FFGS_HOME"]).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache) / "ffgs"


def hub_cache() -> Path:
    """The Hugging Face cache the weights are in (`HF_HUB_CACHE`)."""
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


# --- records ------------------------------------------------------------------------


def resolve(
    source: str | Path, *, revision: str | None = None, subfolder: str | None = None
) -> Record:
    """The record `source` would have (unsaved): its type, and the source as stored.

    Typed like `GSPipeline.from_pretrained` reads it: an existing directory, then a
    zoo name, then a Hub repo id (`owner/name`, checked for form only)."""
    if Path(source).is_dir():
        if revision is not None:
            raise ValueError(f"{source} is a local directory: revision does not apply")
        return Record(str(Path(source).resolve()), "local", None, subfolder)
    name = zoo.resolve(str(source))  # an unknown variant of a zoo family raises
    if name is not None:
        if revision is not None or subfolder is not None:
            raise ValueError(
                f"{name} is a zoo entry pinned to its own revision and file: "
                "revision and subfolder do not apply"
            )
        return Record(name, "zoo")
    if not _REPO.fullmatch(str(source)):
        raise ValueError(
            f"{source} is not a directory, a zoo name (see `ffgs.zoo.list_models()`) "
            "or a Hub repo id (owner/name)"
        )
    return Record(str(source), "hub", revision, subfolder)


def records() -> list[Record]:
    """Every record, sorted by source; unreadable files are skipped."""
    found = []
    directory = home() / "models"
    for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
        try:
            found.append(_read(path))
        except (OSError, ValueError, TypeError):
            continue
    return sorted(found, key=lambda r: (r.source, r.revision or "", r.subfolder or ""))


def get(
    source: str | Path, *, revision: str | None = None, subfolder: str | None = None
) -> Record | None:
    """The record of `source`, or None."""
    wanted = resolve(source, revision=revision, subfolder=subfolder)
    path = _path(wanted)
    return _read(path) if path.is_file() else None


def add(
    source: str | Path, *, revision: str | None = None, subfolder: str | None = None
) -> Record:
    """Record `source` without downloading it; its record if it has one already."""
    record = resolve(source, revision=revision, subfolder=subfolder)
    existing = get(record.source, revision=revision, subfolder=subfolder)
    if existing is not None:
        return existing
    return _write(replace(record, added=_now()))


def touch(
    source: str | Path, *, revision: str | None = None, subfolder: str | None = None
) -> Record:
    """Record that `source` was just used (by `GSPipeline.from_pretrained`): added
    if new, its commit and `last_used` updated."""
    record = add(source, revision=revision, subfolder=subfolder)
    commit = _cached_commit(record) if record.type == "hub" else None
    return _write(replace(record, commit=commit or record.commit, last_used=_now()))


def pull(
    source: str | Path, *, revision: str | None = None, subfolder: str | None = None
) -> Record:
    """Download the weights (and code) of `source` into the Hugging Face cache,
    recording it if new. A zoo entry's weights are checked against its sha256;
    a local directory has nothing to download."""
    record = add(source, revision=revision, subfolder=subfolder)
    if record.type == "local":
        if not _local_dir(record).is_dir():
            raise FileNotFoundError(f"{_local_dir(record)} does not exist")
        return record
    if record.type == "zoo":
        zoo.download_weights(zoo.get_entry(record.source))
        return record
    from huggingface_hub import snapshot_download

    prefix = f"{record.subfolder}/" if record.subfolder else ""
    snapshot = Path(
        snapshot_download(
            record.source,
            revision=record.revision,
            allow_patterns=[prefix + name for name in PIPELINE_FILES],
        )
    )
    if not (snapshot / prefix / "config.json").is_file():
        raise FileNotFoundError(
            f"{record.source}{'/' + record.subfolder if record.subfolder else ''} "
            "has no config.json: not a saved ffgs pipeline"
        )
    return _write(replace(record, commit=snapshot.name))


def status(record: Record) -> Status:
    """Whether the record's files are on disk now, and their size."""
    if record.type == "local":
        directory = _local_dir(record)
        if not directory.is_dir():
            return Status("missing")
        return Status("local", sum(_size(p) for p in directory.rglob("*")))
    try:
        files = _files(record)
    except KeyError:  # a zoo entry this version of ffgs does not have
        return Status("unknown")
    if not files:
        return Status("not cached")
    return Status("cached", sum(_size(p) for p in files))


def remove(source: str | Path) -> list[Record]:
    """Delete the records of `source`, every revision and subfolder of it; the
    records deleted. Records only: files in the Hugging Face cache are left to its
    own tools (`hf cache rm`), and a local directory is the user's. A local
    directory is matched by its absolute path, so a deleted one can be removed."""
    local = str(Path(source).expanduser().resolve())
    found = [
        r
        for r in records()
        if r.source == str(source) or (r.type == "local" and r.source == local)
    ]
    if not found:
        raise KeyError(f"{source} has no record (see `ffgs.store.records()`)")
    for record in found:
        _path(record).unlink(missing_ok=True)
    return found


# --- files in the Hugging Face cache ------------------------------------------------


def _repo_dir(repo: str) -> Path:
    return hub_cache() / ("models--" + repo.replace("/", "--"))


def _cached_commit(record: Record) -> str | None:
    """The commit the cache holds for a Hub record's revision (from its refs)."""
    if record.revision is not None and _COMMIT.fullmatch(record.revision):
        return record.revision
    ref = _repo_dir(record.source) / "refs" / (record.revision or "main")
    try:
        commit = ref.read_text().strip()
    except OSError:
        return None
    return commit if _COMMIT.fullmatch(commit) else None


def _files(record: Record) -> list[Path]:
    """The cached files of a zoo or Hub record ([] when not cached); KeyError for
    a zoo entry this version of ffgs does not have."""
    if record.type == "zoo":
        weights = zoo.get_entry(record.source).weights
        path = _repo_dir(weights.repo) / "snapshots" / weights.revision / weights.file
        return [path] if path.exists() else []
    commit = record.commit or _cached_commit(record)
    if commit is None:
        return []
    snapshot = _repo_dir(record.source) / "snapshots" / commit
    root = snapshot / record.subfolder if record.subfolder else snapshot
    if not (root / "config.json").exists():
        return []
    # A repo's root pipeline is its snapshot less the subfolders other records use.
    claimed = set()
    if record.subfolder is None:
        for other in records():
            if other.source == record.source and other.subfolder is not None:
                claimed.add(snapshot / other.subfolder)
    return sorted(
        p
        for p in root.rglob("*")
        if p.exists()
        and not p.is_dir()
        and not any(c == p or c in p.parents for c in claimed)
    )


def _size(path: Path) -> int:
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


# --- record files -------------------------------------------------------------------


def _local_dir(record: Record) -> Path:
    path = Path(record.source)
    return path / record.subfolder if record.subfolder else path


def _path(record: Record) -> Path:
    key = json.dumps(list(record.key))
    return home() / "models" / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"


def _read(path: Path) -> Record:
    data = json.loads(path.read_text())
    record = Record(**data)
    if record.type not in TYPES:
        raise ValueError(f"{path}: type must be one of {TYPES}")
    return record


def _write(record: Record) -> Record:
    """Atomically: a reader sees the old record or the new one, never a part."""
    path = _path(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(asdict(record), indent=1) + "\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return record


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")
