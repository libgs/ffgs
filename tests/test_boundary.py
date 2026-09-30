import subprocess
import sys


def test_import_does_not_pull_in_gsplat_training_stacks_or_repo_code() -> None:
    code = (
        "import sys; import ffgs; from ffgs import *; "
        "import ffgs.dynamic, ffgs.geometry, ffgs.image, ffgs.pipeline, "
        "ffgs.processor, ffgs.registry, ffgs.render, ffgs.types; "
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in "
        "('gsplat', 'lightning', 'pytorch_lightning', 'omegaconf', "
        "'ffgs_remote_code')); "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_version_is_the_installed_package_version() -> None:
    from importlib.metadata import version

    import ffgs

    assert ffgs.__version__ == version("ffgs")
