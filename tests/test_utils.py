"""Env detection + default input/output paths (kaggle/colab/local)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.utils import (  # noqa: E402
    default_analysis_output,
    default_runs_base,
    default_smoke_dir,
    detect_runtime_env,
    is_bf16_available,
    resolve_compute_dtype,
)


class TestDetectRuntimeEnv:
    def test_local_by_default(self, monkeypatch):
        monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
        monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        monkeypatch.delitem(sys.modules, "google.colab", raising=False)
        assert detect_runtime_env() == "local"

    def test_kaggle_env_var_wins(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "notebook")
        assert detect_runtime_env() == "kaggle"

    def test_colab_env_var_wins(self, monkeypatch):
        monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
        monkeypatch.setenv("COLAB_RELEASE_TAG", "release")
        assert detect_runtime_env() == "colab"

    def test_kaggle_beats_colab(self, monkeypatch):
        monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "notebook")
        monkeypatch.setenv("COLAB_RELEASE_TAG", "release")
        assert detect_runtime_env() == "kaggle"


class TestDefaultPaths:
    def test_explicit_env_needs_no_fs(self):
        assert default_runs_base("kaggle") == Path("/kaggle/working/runs")
        assert default_analysis_output("kaggle") == Path("/kaggle/working/analysis")
        assert default_smoke_dir("kaggle") == Path("/kaggle/working/smoke")
        assert default_runs_base("local") == Path(".omo/evidence/runs")
        assert default_analysis_output("local") == Path(".omo/evidence/analysis")
        assert default_smoke_dir("local") == Path(".omo/evidence/smoke")

    def test_colab_without_drive(self, monkeypatch):
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        assert default_runs_base("colab") == Path("/content/runs")
        assert default_analysis_output("colab") == Path("/content/analysis")
        assert default_smoke_dir("colab") == Path("/content/smoke")

    def test_colab_with_drive_mount(self, monkeypatch):
        monkeypatch.setattr(
            "pathlib.Path.exists",
            lambda self: str(self).startswith("/content/drive"),
        )
        assert default_runs_base("colab") == Path(
            "/content/drive/MyDrive/spectral-joint-policy/runs"
        )


class TestBuilderIncludesUtils:
    def builder(self):
        import importlib.util

        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "build_notebook", root / "scripts" / "build_notebook.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_notebooks_inline_no_source(self):
        b = self.builder()
        for spectral in (True, False):
            nb = b.build_notebook(None, spectral=spectral)
            code = "\n".join(
                c["source"] for c in nb.cells if c["cell_type"] == "code"
            )
            assert "Register inlined" not in code
            assert "sys.modules['sjfpo" not in code
            assert "class StateEncoder" not in code
            assert "scripts/train.py" in code

    def test_clone_cell_uses_token_url(self):
        b = self.builder()
        assert "GITHUB_TOKEN" in b.CLONE_CELL
        assert b.REPO_SLUG in b.CLONE_CELL
        assert "getpass" in b.CLONE_CELL


class TestBf16AutoDetect:
    def test_detector_agrees_with_dtype(self):
        import torch

        avail = is_bf16_available()
        assert isinstance(avail, bool)
        assert resolve_compute_dtype() == (torch.bfloat16 if avail else torch.float32)
