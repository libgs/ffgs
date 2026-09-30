"""Models that ffgs does not ship: installed plugins and code in the model repo."""

import importlib
import json
import sys
import textwrap
from pathlib import Path

import huggingface_hub
import huggingface_hub.hub_mixin
import pytest
import torch
from _dummy import MODEL_TYPE, DummyModel, DummyProcessor
from test_pipeline import _model, _views

from ffgs import GSPipeline, ModelFrame, Processor, dynamic, get_model_spec, registry
from ffgs.registry import ENTRY_POINT_GROUP, spec_for_model

PLUGIN_TYPE = "ffgs-test-plugin"
REMOTE_TYPE = "ffgs-test-remote"


@pytest.fixture
def clean_registry():
    """Forget model types registered by a test, and the plugin / built-in model
    modules it imported (a built-in registers on import, so it must import again)."""
    before = dict(registry._REGISTRY)
    modules = set(sys.modules)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(before)
    for name in set(sys.modules) - modules:
        if name.startswith(("ffgs_test_plugin", "ffgs.models.")):
            del sys.modules[name]


def _install_plugin(root: Path, monkeypatch, entry_points: str, code: str) -> None:
    """A distribution on sys.path with `ffgs.models` entry points."""
    (root / "ffgs_test_plugin").mkdir(parents=True)
    (root / "ffgs_test_plugin" / "__init__.py").write_text(textwrap.dedent(code))
    dist = root / "ffgs_test_plugin-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: ffgs-test-plugin\nVersion: 0.1\n"
    )
    (dist / "entry_points.txt").write_text(
        f"[{ENTRY_POINT_GROUP}]\n" + textwrap.dedent(entry_points)
    )
    monkeypatch.syspath_prepend(str(root))
    importlib.invalidate_caches()


PLUGIN_CODE = """
    from _dummy import DummyModel, DummyProcessor
    from ffgs import ModelSpec

    class PluginModel(DummyModel):
        pass

    SPEC = ModelSpec("ffgs-test-plugin", PluginModel, DummyProcessor)
"""


def test_plugin_is_discovered_through_its_entry_point(
    tmp_path, monkeypatch, clean_registry
) -> None:
    _install_plugin(
        tmp_path / "site",
        monkeypatch,
        f"{PLUGIN_TYPE} = ffgs_test_plugin:SPEC\n",
        PLUGIN_CODE,
    )
    assert PLUGIN_TYPE in registry.known_model_types()
    assert "ffgs_test_plugin" not in sys.modules  # listed, not imported

    GSPipeline(_model(), DummyProcessor(), model_type=PLUGIN_TYPE).save_pretrained(
        tmp_path / "saved"
    )
    loaded = GSPipeline.from_pretrained(tmp_path / "saved")
    assert "ffgs_test_plugin" in sys.modules
    assert type(loaded.model).__name__ == "PluginModel"
    assert loaded.model_type == PLUGIN_TYPE
    loaded(_views())
    # A pipeline built from a plugin model finds its model_type too.
    assert spec_for_model(loaded.model).model_type == PLUGIN_TYPE


def test_plugin_module_that_registers_itself(
    tmp_path, monkeypatch, clean_registry
) -> None:
    code = PLUGIN_CODE + "\n    from ffgs import register_model\n"
    code += "    register_model(SPEC)\n"
    _install_plugin(tmp_path, monkeypatch, f"{PLUGIN_TYPE} = ffgs_test_plugin\n", code)
    assert get_model_spec(PLUGIN_TYPE).model_cls.__name__ == "PluginModel"


def test_spec_for_model_discovers_plugins(
    tmp_path, monkeypatch, clean_registry
) -> None:
    _install_plugin(
        tmp_path, monkeypatch, f"{PLUGIN_TYPE} = ffgs_test_plugin:SPEC\n", PLUGIN_CODE
    )
    from ffgs_test_plugin import PluginModel  # imported, not registered

    assert PLUGIN_TYPE not in registry._REGISTRY
    assert GSPipeline(PluginModel(), DummyProcessor()).model_type == PLUGIN_TYPE


@pytest.mark.parametrize(
    ("entry_point", "error", "match"),
    [
        ("other-name = ffgs_test_plugin:SPEC", ValueError, "must be the model_type"),
        ("other-name = ffgs_test_plugin", TypeError, "must be a ModelSpec"),
    ],
)
def test_malformed_plugins_are_reported(
    tmp_path, monkeypatch, clean_registry, entry_point, error, match
) -> None:
    _install_plugin(tmp_path, monkeypatch, entry_point + "\n", PLUGIN_CODE)
    with pytest.raises(error, match=match):
        get_model_spec("other-name")


def test_built_in_and_registered_models_come_before_plugins(
    tmp_path, monkeypatch, clean_registry
) -> None:
    # A plugin claiming an already registered model_type is never imported.
    _install_plugin(
        tmp_path, monkeypatch, f"{MODEL_TYPE} = ffgs_test_plugin:SPEC\n", PLUGIN_CODE
    )
    assert get_model_spec(MODEL_TYPE).model_cls is DummyModel
    assert "ffgs_test_plugin" not in sys.modules


def test_unknown_model_type_says_how_to_add_it(tmp_path) -> None:
    GSPipeline(
        _model(), DummyProcessor(), model_type="nobody-ships-this"
    ).save_pretrained(tmp_path)
    with pytest.raises(KeyError) as info:
        GSPipeline.from_pretrained(tmp_path)
    message = str(info.value)
    for hint in (
        "'nobody-ships-this'",
        MODEL_TYPE,  # the known types are listed
        ENTRY_POINT_GROUP,
        "register_model",
        "auto_map",
        "trust_remote_code=True",
    ):
        assert hint in message


# --- code shipped in the repo (config.json:auto_map) ------------------------------

REMOTE_FILES = {
    "layers.py": """
        from torch import nn


        def make_proj(num_gaussians, colors):
            return nn.Linear(6, num_gaussians * (11 + colors))
    """,
    "modeling_remote.py": """
        from pathlib import Path

        import torch
        from huggingface_hub import PyTorchModelHubMixin
        from torch import nn

        from .layers import make_proj

        # Leaves a trace, so tests can tell whether this file ever ran.
        (Path(__file__).parent / "ran.txt").write_text("imported")


        class RemoteModel(nn.Module, PyTorchModelHubMixin):
            def __init__(self, num_gaussians=5, sh_degree=1):
                super().__init__()
                self.num_gaussians = num_gaussians
                self.proj = make_proj(num_gaussians, 3 * (sh_degree + 1) ** 2)

            def forward(self, model_input):
                pooled = model_input["images"].mean(dim=(1, 3, 4))
                centres = model_input["c2w"][:, :, :3, 3].mean(dim=1)
                out = self.proj(torch.cat((pooled, centres), dim=-1))
                return out.reshape(out.shape[0], self.num_gaussians, -1)
    """,
    "processing_remote.py": """
        import torch

        from ffgs import Gaussians, Prepared, Processor
        from ffgs.geometry import anchored_frame
        from ffgs.processor import ImageFitConfig


        class RemoteProcessor(Processor):
            config_cls = ImageFitConfig

            def preprocess(self, views, device):
                frame = anchored_frame(views.c2w, 0.2)
                fitted = self.fit_views(views)
                return Prepared(
                    {
                        "images": fitted.images.float().to(device),
                        "c2w": frame.c2w_to_model(fitted.c2w.float()).to(device),
                    },
                    frame,
                )

            def postprocess(self, packed):
                b, n, _ = packed.shape
                return Gaussians(
                    means=packed[..., 0:3],
                    opacities=torch.sigmoid(packed[..., 3]),
                    scales=torch.exp(packed[..., 4:7]),
                    quats=torch.nn.functional.normalize(packed[..., 7:11], dim=-1),
                    colors=packed[..., 11:].reshape(b, n, 4, 3),
                    sh_degree=1,
                )

            def render_planes(self, frame):
                return 0.01, 100.0
    """,
}
AUTO_MAP = {
    "model": "modeling_remote.RemoteModel",
    "processor": "processing_remote.RemoteProcessor",
}


def _remote_repo(path: Path, **config) -> Path:
    """A model directory whose classes live in its own .py files."""
    path.mkdir(parents=True)
    for name, code in REMOTE_FILES.items():
        (path / name).write_text(textwrap.dedent(code))
    # Same layers as the dummy model, so its weights load.
    _model().save_pretrained(path)
    config = {
        "num_gaussians": 5,
        "sh_degree": 1,
        "model_type": REMOTE_TYPE,
        "auto_map": AUTO_MAP,
        **config,
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "processor_config.json").write_text(json.dumps({"image_shape": [16, 32]}))
    return path


def test_repo_code_is_refused_without_trust_remote_code(tmp_path) -> None:
    repo = _remote_repo(tmp_path / "repo")
    with pytest.raises(ValueError, match="trust_remote_code=True") as info:
        GSPipeline.from_pretrained(repo)
    assert "modeling_remote.RemoteModel" in str(info.value)
    assert not (repo / "ran.txt").exists()  # nothing from the repo ran
    assert not any(  # nor was the directory imported as a package
        list(module.__path__) == [str(repo.resolve())]
        for name, module in sys.modules.items()
        if name.startswith("ffgs_remote_code.")
    )


def test_repo_code_loads_with_trust_remote_code(tmp_path) -> None:
    repo = _remote_repo(tmp_path / "repo")
    pipe = GSPipeline.from_pretrained(repo, trust_remote_code=True)
    assert (repo / "ran.txt").exists()
    assert type(pipe.model).__name__ == "RemoteModel"
    assert type(pipe.processor).__name__ == "RemoteProcessor"
    assert pipe.model_type == REMOTE_TYPE
    assert REMOTE_TYPE not in registry._REGISTRY  # not registered process-wide

    # Same layers and weights as the dummy model -> same Gaussians.
    reference = GSPipeline(_model(), DummyProcessor(image_shape=[16, 32]))
    views = _views()
    out, expected = pipe(views), reference(views)
    assert torch.allclose(out.gaussians.means, expected.gaussians.means)

    # Saving keeps the code with the weights; the copy loads on its own.
    saved = pipe.save_pretrained(tmp_path / "saved")
    config = json.loads((saved / "config.json").read_text())
    assert config["auto_map"] == AUTO_MAP
    assert config["model_type"] == REMOTE_TYPE
    for name in REMOTE_FILES:
        assert (saved / name).read_text() == (repo / name).read_text()
    with pytest.raises(ValueError, match="trust_remote_code"):
        GSPipeline.from_pretrained(saved)
    again = GSPipeline.from_pretrained(saved, trust_remote_code=True)
    assert torch.equal(again(views).gaussians.means, out.gaussians.means)
    # Re-saving onto the directory it came from leaves the code intact.
    again.save_pretrained(saved)
    assert (saved / "modeling_remote.py").read_text() == (
        repo / "modeling_remote.py"
    ).read_text()


def test_saved_repo_code_includes_modules_imported_when_run(tmp_path) -> None:
    repo = _remote_repo(tmp_path / "repo")
    # A helper the model imports only when it runs.
    (repo / "lazy.py").write_text(
        "def pool(images):\n    return images.mean(dim=(1, 3, 4))\n"
    )
    modeling = repo / "modeling_remote.py"
    code = modeling.read_text()
    eager = 'pooled = model_input["images"].mean(dim=(1, 3, 4))'
    assert eager in code
    lazy = 'from .lazy import pool\n\n        pooled = pool(model_input["images"])'
    modeling.write_text(code.replace(eager, lazy))
    pipe = GSPipeline.from_pretrained(repo, trust_remote_code=True)
    saved = pipe.save_pretrained(tmp_path / "saved")  # before forward imports lazy
    assert (saved / "lazy.py").read_text() == (repo / "lazy.py").read_text()
    again = GSPipeline.from_pretrained(saved, trust_remote_code=True)
    assert torch.equal(again(_views()).gaussians.means, pipe(_views()).gaussians.means)


def test_installed_models_come_before_repo_code(tmp_path) -> None:
    repo = _remote_repo(tmp_path / "repo", model_type=MODEL_TYPE)
    pipe = GSPipeline.from_pretrained(repo)  # no trust needed: code is not used
    assert type(pipe.model) is DummyModel
    assert not (repo / "ran.txt").exists()


@pytest.mark.parametrize(
    ("auto_map", "match"),
    [
        ({"model": AUTO_MAP["model"]}, "must map exactly"),
        ({**AUTO_MAP, "model": "RemoteModel"}, "module.Class"),
        ({**AUTO_MAP, "model": "someone/code--modeling.Model"}, "another repo"),
        ({**AUTO_MAP, "model": "modeling_missing.Model"}, "is not in"),
        ({**AUTO_MAP, "model": "layers.Missing"}, "no class 'Missing'"),
        ({**AUTO_MAP, "processor": "modeling_remote.RemoteModel"}, "ffgs.Processor"),
    ],
)
def test_malformed_auto_map_is_reported(tmp_path, auto_map, match) -> None:
    repo = _remote_repo(tmp_path / "repo", auto_map=auto_map)
    with pytest.raises((ValueError, ImportError, TypeError), match=match):
        GSPipeline.from_pretrained(repo, trust_remote_code=True)


# --- shipping the code of an installed package (include_code=True) ---------------

_REMOTE = {name: textwrap.dedent(code) for name, code in REMOTE_FILES.items()}
PACKAGE_FILES = {
    "acme_gs/__init__.py": "",
    "acme_gs/other.py": "VALUE = 1\n",
    "acme_gs/models/__init__.py": "",
    "acme_gs/models/dummy/__init__.py": "from .modeling import AcmeModel\n",
    "acme_gs/models/dummy/nn/__init__.py": "",
    "acme_gs/models/dummy/nn/layers.py": _REMOTE["layers.py"],
    "acme_gs/models/dummy/modeling.py": _REMOTE["modeling_remote.py"]
    .replace("RemoteModel", "AcmeModel")
    .replace("from .layers", "from .nn.layers"),
    "acme_gs/models/dummy/processing.py": _REMOTE["processing_remote.py"].replace(
        "RemoteProcessor", "AcmeProcessor"
    ),
}
SHIPPED = {
    name.removeprefix("acme_gs/models/dummy/"): code
    for name, code in PACKAGE_FILES.items()
    if name.startswith("acme_gs/models/dummy/")
}


def _install_package(root: Path, monkeypatch, files: dict[str, str]):
    for name, code in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(code)
    monkeypatch.syspath_prepend(str(root))
    importlib.invalidate_caches()
    modeling = importlib.import_module("acme_gs.models.dummy.modeling")
    processing = importlib.import_module("acme_gs.models.dummy.processing")
    return modeling.AcmeModel, processing.AcmeProcessor


def _uninstall_package(root: Path) -> None:
    sys.path.remove(str(root))
    for name in [m for m in sys.modules if m.split(".")[0] == "acme_gs"]:
        del sys.modules[name]
    importlib.invalidate_caches()


def test_installed_code_ships_with_include_code(tmp_path, monkeypatch) -> None:
    root = tmp_path / "site"
    model_cls, processor_cls = _install_package(root, monkeypatch, PACKAGE_FILES)
    pipe = GSPipeline(
        model_cls(), processor_cls(image_shape=[16, 32]), model_type="acme"
    )
    views = _views()
    expected = pipe(views).gaussians.means

    plain = pipe.save_pretrained(tmp_path / "plain")
    assert "auto_map" not in json.loads((plain / "config.json").read_text())
    assert not list(plain.rglob("*.py"))

    saved = pipe.save_pretrained(tmp_path / "saved", include_code=True)
    config = json.loads((saved / "config.json").read_text())
    assert config["model_type"] == "acme"
    assert config["auto_map"] == {
        "model": "modeling.AcmeModel",
        "processor": "processing.AcmeProcessor",
    }
    # The innermost package holding both classes, subpackages included.
    shipped = {str(p.relative_to(saved)) for p in saved.rglob("*.py")}
    assert shipped == set(SHIPPED)
    for name, code in SHIPPED.items():
        assert (saved / name).read_text() == code

    # Loads where the package is not installed, from the shipped code only.
    _uninstall_package(root)
    with pytest.raises(ValueError, match="trust_remote_code"):
        GSPipeline.from_pretrained(saved)
    loaded = GSPipeline.from_pretrained(saved, trust_remote_code=True)
    assert type(loaded.model).__module__.startswith("ffgs_remote_code.")
    assert not any(m.split(".")[0] == "acme_gs" for m in sys.modules)
    assert torch.equal(loaded(views).gaussians.means, expected)


def test_code_importing_outside_its_package_is_refused(tmp_path, monkeypatch) -> None:
    files = dict(PACKAGE_FILES)
    files[
        "acme_gs/models/dummy/nn/layers.py"
    ] += "\nfrom acme_gs.other import VALUE\nfrom ....other import VALUE as V2\n"
    root = tmp_path / "site"
    model_cls, processor_cls = _install_package(root, monkeypatch, files)
    pipe = GSPipeline(model_cls(), processor_cls(), model_type="acme")
    with pytest.raises(ValueError, match="relatively") as info:
        pipe.save_pretrained(tmp_path / "saved", include_code=True)
    message = str(info.value)
    assert "nn/layers.py:8: acme_gs.other" in message
    assert "nn/layers.py:9: ....other" in message
    assert not (tmp_path / "saved").exists()  # nothing written
    _uninstall_package(root)


def test_code_must_be_one_package_outside_ffgs() -> None:
    with pytest.raises(ValueError, match="one package"):
        dynamic.export_code(DummyModel, DummyProcessor, Path("unused"))
    with pytest.raises(ValueError, match="built into ffgs"):
        dynamic.export_code(ModelFrame, Processor, Path("unused"))


# --- Hub options ------------------------------------------------------------------


def _fake_hub(monkeypatch, repos: dict[str, Path]) -> list[dict]:
    """Serve repo ids from local directories; record every download's options."""
    calls = []

    def hf_hub_download(repo_id, filename, **kwargs):
        calls.append(kwargs)
        return str(repos[repo_id] / filename)

    def snapshot_download(repo_id, **kwargs):
        calls.append(kwargs)
        return str(repos[repo_id])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(huggingface_hub.hub_mixin, "hf_hub_download", hf_hub_download)
    return calls


HUB_OPTIONS = {
    "revision": "0123abcd",
    "cache_dir": "/tmp/nowhere",
    "token": "hf_x",
    "force_download": True,
    "local_files_only": False,
}


def test_hub_options_reach_every_download(tmp_path, monkeypatch) -> None:
    saved = tmp_path / "saved"
    GSPipeline(_model(), DummyProcessor()).save_pretrained(saved)
    repo = _remote_repo(tmp_path / "remote")
    calls = _fake_hub(monkeypatch, {"someone/dummy": saved, "someone/remote": repo})

    GSPipeline.from_pretrained("someone/dummy", **HUB_OPTIONS)
    # config.json, model config, weights, processor_config.json
    assert len(calls) == 4
    GSPipeline.from_pretrained("someone/remote", trust_remote_code=True, **HUB_OPTIONS)
    assert len(calls) == 9  # + the code snapshot
    assert calls[5]["allow_patterns"] == ["*.py"]
    for call in calls:
        for key, value in HUB_OPTIONS.items():
            assert call[key] == value


def test_unknown_options_are_rejected(tmp_path) -> None:
    GSPipeline(_model(), DummyProcessor()).save_pretrained(tmp_path)
    with pytest.raises(TypeError, match="revison"):
        GSPipeline.from_pretrained(tmp_path, revison="v1")
