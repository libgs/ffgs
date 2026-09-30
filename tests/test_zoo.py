"""The model zoo: entries shipped in `ffgs/zoo`, loaded by name from a fake Hub."""

import copy
import json
import subprocess
import sys

import pytest
import torch
from _dummy import DummyModel, DummyProcessor
from safetensors.torch import save_file
from test_pipeline import _model, _views
from test_third_party import _fake_hub

from ffgs import GSPipeline, ModelSpec, get_model_spec, register_model, zoo

UPSTREAM_TYPE = "ffgs-test-upstream"
RAW_TYPE = "ffgs-test-raw"
REPO = "author/upstream"
COMMIT = "0123456789abcdef0123456789abcdef01234567"


class UpstreamModel(DummyModel):
    pass


def _from_upstream(checkpoint):
    """The upstream checkpoint nests its weights under `net.` in `state_dict`."""
    return {k.removeprefix("net."): v for k, v in checkpoint["state_dict"].items()}


@pytest.fixture
def upstream(tmp_path, monkeypatch, clean_registry):
    """A zoo with one family, pointing at an upstream checkpoint on a fake Hub."""
    register_model(
        ModelSpec(UPSTREAM_TYPE, UpstreamModel, DummyProcessor, _from_upstream)
    )
    register_model(ModelSpec(RAW_TYPE, UpstreamModel, DummyProcessor))
    model = _model()
    repo = tmp_path / "hub" / "upstream"
    (repo / "ckpts").mkdir(parents=True)
    checkpoint = {"state_dict": {f"net.{k}": v for k, v in model.state_dict().items()}}
    torch.save(checkpoint, repo / "ckpts" / "dummy.pt")
    root = tmp_path / "zoo"
    (root / "dummy").mkdir(parents=True)
    (root / "dummy" / "notes.txt").write_text("not an entry")
    monkeypatch.setattr(zoo, "_ROOT", root)
    calls = _fake_hub(monkeypatch, {REPO: repo})

    def write(variant="v1", **changes):
        entry = _entry(zoo.sha256_file(repo / "ckpts" / "dummy.pt"))
        for key, value in changes.items():
            if value is None:
                del entry[key]
            else:
                entry[key] = value
        (root / "dummy" / f"{variant}.json").write_text(json.dumps(entry))
        return entry

    write()
    return {"model": model, "repo": repo, "calls": calls, "write": write}


def _entry(sha256: str) -> dict:
    return {
        "model_type": UPSTREAM_TYPE,
        "description": "A dummy model in an upstream layout",
        "weights": {
            "repo": REPO,
            "file": "ckpts/dummy.pt",
            "revision": COMMIT,
            "sha256": sha256,
        },
        "model": {"num_gaussians": 5, "sh_degree": 1},
        "processor": {"image_shape": [16, 32], "scene_scale": 0.3},
        "source": {"url": "https://example.com/upstream", "paper": "https://x.org"},
        "license": "Apache-2.0",
    }


def _same_weights(a: torch.nn.Module, b: torch.nn.Module) -> None:
    assert a.state_dict().keys() == b.state_dict().keys()
    for key, value in a.state_dict().items():
        assert torch.equal(b.state_dict()[key], value), key


# --- listing and names ------------------------------------------------------------


def test_listing_reads_directory_names_only(upstream) -> None:
    upstream["write"]("v2")
    assert zoo.families() == ["dummy"]
    assert zoo.list_models() == ["dummy/v1", "dummy/v2"]
    assert zoo.list_models("dummy") == ["dummy/v1", "dummy/v2"]
    assert zoo.list_models("missing") == []


def test_names_resolve_local_directory_then_zoo_then_hub(
    upstream, tmp_path, monkeypatch
) -> None:
    assert zoo.resolve("dummy/v1") == "dummy/v1"
    assert zoo.resolve("someone/repo") is None  # not a zoo family: a Hub repo
    assert zoo.resolve(tmp_path) is None  # a Path is never a zoo name
    with pytest.raises(KeyError, match=r"dummy/v3.*\['dummy/v1'\].*snapshot_download"):
        zoo.resolve("dummy/v3")
    with pytest.raises(KeyError, match="dummy/v3"):
        GSPipeline.from_pretrained("dummy/v3")
    for name in ("dummy/../dummy/v1", "dummy/v1/x"):
        with pytest.raises(KeyError, match="no zoo entry"):
            zoo.resolve(name)
        with pytest.raises(KeyError, match="no zoo entry"):
            zoo.get_entry(name)
    assert upstream["calls"] == []  # a reserved family never reaches the Hub

    # A local directory of the same name wins over the zoo.
    monkeypatch.chdir(tmp_path)
    GSPipeline(_model(), DummyProcessor(scene_scale=0.7)).save_pretrained("dummy/v1")
    assert zoo.resolve("dummy/v1") is None
    loaded = GSPipeline.from_pretrained("dummy/v1")
    assert loaded.processor.config.scene_scale == 0.7
    assert upstream["calls"] == []


# --- loading ----------------------------------------------------------------------


def test_zoo_entry_loads_pinned_checked_and_converted(upstream, tmp_path) -> None:
    pipe = GSPipeline.from_pretrained("dummy/v1", cache_dir="/tmp/c", token="hf_x")
    (call,) = upstream["calls"]
    assert call == {
        "repo_id": REPO,
        "filename": "ckpts/dummy.pt",
        "revision": COMMIT,
        "cache_dir": "/tmp/c",
        "token": "hf_x",
    }
    assert type(pipe.model) is UpstreamModel
    assert pipe.model_type == UPSTREAM_TYPE
    assert pipe.processor.config.scene_scale == 0.3
    _same_weights(upstream["model"], pipe.model)
    reference = GSPipeline(upstream["model"], pipe.processor, UPSTREAM_TYPE)
    views = _views()
    assert torch.equal(pipe(views).gaussians.means, reference(views).gaussians.means)

    # Saved, it is an ordinary ffgs directory (no conversion needed any more).
    pipe.save_pretrained(tmp_path / "saved")
    config = json.loads((tmp_path / "saved" / "config.json").read_text())
    assert config["model_type"] == UPSTREAM_TYPE
    assert config["num_gaussians"] == 5
    reloaded = GSPipeline.from_pretrained(tmp_path / "saved")
    _same_weights(pipe.model, reloaded.model)
    assert reloaded.processor.config == pipe.processor.config


def test_zoo_entry_takes_processor_overrides_and_safetensors(upstream) -> None:
    model = upstream["model"]
    save_file(model.state_dict(), upstream["repo"] / "flat.safetensors")
    weights = {
        "repo": REPO,
        "file": "flat.safetensors",
        "revision": COMMIT,
        "sha256": zoo.sha256_file(upstream["repo"] / "flat.safetensors"),
    }
    upstream["write"]("flat", model_type=RAW_TYPE, weights=weights)
    pipe = GSPipeline.from_pretrained(
        "dummy/flat", processor_overrides={"crop_mode": "pad"}
    )
    _same_weights(model, pipe.model)
    assert pipe.processor.config.crop_mode == "pad"
    assert pipe.processor.config.scene_scale == 0.3


def test_sha256_mismatch_is_an_error(upstream) -> None:
    upstream["write"](weights={**_entry("0" * 64)["weights"]})
    with pytest.raises(ValueError, match="sha256 .* expected 0{64}.*force_download"):
        GSPipeline.from_pretrained("dummy/v1")


def test_weights_must_fit_the_model_exactly(upstream) -> None:
    # Without the model type's conversion the upstream layout does not fit.
    upstream["write"](model_type=RAW_TYPE)
    match = r"(?s)dummy/v1.*convert_state_dict.*Missing key.*proj.weight"
    with pytest.raises(RuntimeError, match=match):
        GSPipeline.from_pretrained("dummy/v1")
    # A model config that does not match the checkpoint's shapes.
    upstream["write"](model={"num_gaussians": 4, "sh_degree": 1})
    with pytest.raises(RuntimeError, match="size mismatch"):
        GSPipeline.from_pretrained("dummy/v1")


def test_checkpoints_are_unpickled_as_plain_tensors_only(upstream) -> None:
    path = upstream["repo"] / "ckpts" / "dummy.pt"
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["args"] = UpstreamModel  # arbitrary pickled objects are refused
    torch.save(checkpoint, path)
    upstream["write"]()
    with pytest.raises(Exception, match="[Ww]eights only load failed"):
        GSPipeline.from_pretrained("dummy/v1")


def test_zoo_names_take_no_revision_or_conflicting_model_type(upstream) -> None:
    with pytest.raises(TypeError, match=r"pinned .*\['revision'\]"):
        GSPipeline.from_pretrained("dummy/v1", revision="main")
    with pytest.raises(TypeError, match=r"\['subfolder'\]"):
        GSPipeline.from_pretrained("dummy/v1", subfolder="x")
    with pytest.raises(ValueError, match="is a 'ffgs-test-upstream' model"):
        GSPipeline.from_pretrained("dummy/v1", model_type="other")
    assert upstream["calls"] == []


def test_unknown_processor_keys_in_an_entry_are_reported(upstream) -> None:
    upstream["write"](processor={"scene_scal": 0.3})
    with pytest.raises(ValueError, match=r"dummy/v1: processor .*\['scene_scal'\]"):
        GSPipeline.from_pretrained("dummy/v1")


# --- schema -----------------------------------------------------------------------

_VALID = _entry("a" * 64)


def _with(path: str, value):
    """_VALID with `path` ("key" or "key.sub") set to `value` (None: removed)."""
    entry = copy.deepcopy(_VALID)
    *parents, key = path.split(".")
    target = entry
    for parent in parents:
        target = target[parent]
    if value is None:
        del target[key]
    else:
        target[key] = value
    return entry


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ([], "JSON object"),
        (_with("license", None), r"missing keys \['license'\]"),
        (_with("weight", {}), r"unknown keys \['weight'\]"),
        (_with("model", [1]), "model must be a JSON object"),
        (_with("weights", "author/upstream"), "weights must be a JSON object"),
        (_with("weights.sha256", None), r"missing keys \['weights.sha256'\]"),
        (_with("weights.subfolder", "x"), r"unknown keys \['weights.subfolder'\]"),
        (_with("weights.revision", "main"), "full 40-hex commit"),
        (_with("weights.revision", "0123abcd"), "full 40-hex commit"),
        (_with("weights.sha256", "A" * 64), "64 lowercase hex"),
        (_with("weights.repo", "upstream"), "'owner/name'"),
        (_with("weights.file", ""), "weights.file must be a non-empty string"),
        (_with("source.url", None), r"missing keys \['source.url'\]"),
        (_with("source.code", "x"), r"unknown keys \['source.code'\]"),
        (_with("source.paper", 1), "non-empty strings"),
    ],
)
def test_malformed_entries_are_rejected(entry, match) -> None:
    with pytest.raises(ValueError, match=f"zoo entry f/v: .*{match}"):
        zoo.parse_entry("f/v", entry)


def test_a_valid_entry_parses() -> None:
    entry = zoo.parse_entry("f/v", _VALID)
    assert entry.weights == zoo.Weights(REPO, "ckpts/dummy.pt", COMMIT, "a" * 64)
    assert entry.source == {
        "url": "https://example.com/upstream",
        "paper": "https://x.org",
    }
    assert zoo.parse_entry("f/v", _with("source.paper", None)).source.keys() == {"url"}


def test_invalid_json_is_reported(upstream) -> None:
    (zoo._ROOT / "dummy" / "v1.json").write_text("{")
    with pytest.raises(ValueError, match="dummy/v1: invalid JSON"):
        GSPipeline.from_pretrained("dummy/v1")


# --- subfolder (Hub option for repos with variants in subdirectories) -------------


def test_subfolder_of_a_local_directory_and_a_hub_repo(tmp_path, monkeypatch) -> None:
    family = tmp_path / "family"
    GSPipeline(_model(), DummyProcessor(scene_scale=0.4)).save_pretrained(
        family / "re10k-2v"
    )
    local = GSPipeline.from_pretrained(family, subfolder="re10k-2v")
    assert local.processor.config.scene_scale == 0.4
    with pytest.raises(FileNotFoundError, match="no subfolder 'dl3dv'"):
        GSPipeline.from_pretrained(family, subfolder="dl3dv")

    calls = _fake_hub(monkeypatch, {"someone/family": family})
    hub = GSPipeline.from_pretrained(
        "someone/family", subfolder="re10k-2v", revision="0123abcd"
    )
    _same_weights(local.model, hub.model)
    (call,) = calls
    assert call["revision"] == "0123abcd"
    assert call["allow_patterns"] == [
        "re10k-2v/config.json",
        "re10k-2v/model.safetensors",
        "re10k-2v/pytorch_model.bin",
        "re10k-2v/processor_config.json",
        "re10k-2v/*.py",
    ]


# --- import boundary ----------------------------------------------------------------


def test_import_and_listing_read_no_entry() -> None:
    code = (
        "import sys\n"
        "opened = []\n"
        "def hook(event, args):\n"
        "    if event == 'open' and str(args[0]).endswith('.json'):\n"
        "        opened.append(str(args[0]))\n"
        "sys.addaudithook(hook)\n"
        "import ffgs\n"
        "from ffgs import zoo\n"
        "zoo.list_models()\n"
        "bad = [p for p in opened if 'zoo' in p]\n"
        "print(bad); sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --- the shipped entries ----------------------------------------------------------


@pytest.mark.parametrize("name", zoo.list_models())
def test_shipped_entry_parses_and_builds(name) -> None:
    entry = zoo.get_entry(name)
    spec = get_model_spec(entry.model_type)
    spec.model_cls(**entry.model)
    spec.processor_cls.from_dict(entry.processor)


@pytest.mark.network
@pytest.mark.parametrize("name", zoo.list_models())
def test_shipped_entry_downloads_and_fits(name) -> None:
    # sha256 checked, conversion applied, every key matched (strict load).
    GSPipeline.from_pretrained(name)
