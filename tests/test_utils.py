"""Env detection + default input/output paths (kaggle/colab/local)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sjfpo.utils import (  # noqa: E402
    default_runs_base,
    default_smoke_dir,
    detect_runtime_env,
    resolve_compute_dtype,
    resolve_storage_dtype,
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
        assert default_smoke_dir("kaggle") == Path("/kaggle/working/smoke")
        assert default_runs_base("local") == Path(".omo/evidence/runs")
        assert default_smoke_dir("local") == Path(".omo/evidence/smoke")

    def test_colab_without_drive(self, monkeypatch):
        monkeypatch.setattr("pathlib.Path.exists", lambda self: False)
        assert default_runs_base("colab") == Path("/content/runs")
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
        nb = b.build_notebook()
        code = "\n".join(c["source"] for c in nb.cells if c["cell_type"] == "code")
        assert "Register inlined" not in code
        assert "sys.modules['sjfpo" not in code
        assert "class StateEncoder" not in code
        assert "scripts/train.py" in code

    def test_notebook_writefile_cells_cover_runtime_files(self):
        b = self.builder()
        nb = b.build_notebook()
        heads = [
            str(c["source"]).splitlines()[0].strip()
            for c in nb.cells
            if c["cell_type"] == "code"
        ]
        for p in b.embedded_files():
            assert f"%%writefile {p.relative_to(b.ROOT).as_posix()}" in heads
        assert "!git clone" not in "\n".join(heads)


class TestToDeviceAsync:
    def test_cpu_roundtrip_preserves_values(self):
        import torch
        from sjfpo.utils import to_device_async

        x = torch.randn(4, 5)
        y = to_device_async(x, torch.device("cpu"))
        assert y.device.type == "cpu"
        assert torch.equal(y, x)


class TestBf16AutoDetect:
    def test_detector_agrees_with_dtype(self):
        import torch

        avail = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        assert isinstance(avail, bool)
        assert resolve_compute_dtype() == (torch.bfloat16 if avail else torch.float32)


class TestDtypeResolution:
    def test_cpu_storage_is_fp32_even_on_blackwell(self, monkeypatch):
        import torch
        import sjfpo.utils as utils

        monkeypatch.setattr(utils, "is_rtx_1000_pro_blackwell", lambda: True)
        assert utils.resolve_storage_dtype("cpu") == torch.float32
        assert utils.resolve_storage_dtype("cuda") == torch.bfloat16

    def test_dtype_override_precedes_blackwell_default(self, monkeypatch):
        import torch
        import sjfpo.utils as utils

        monkeypatch.setattr(utils, "cuda_available", lambda: True)
        monkeypatch.setattr(utils, "is_rtx_1000_pro_blackwell", lambda: True)
        monkeypatch.setenv("SJFPO_DTYPE", "fp32")
        assert utils.resolve_compute_dtype() == torch.float32

        monkeypatch.setenv("SJFPO_DTYPE", "bf16")
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
        assert utils.resolve_compute_dtype() == torch.bfloat16

    def test_invalid_dtype_override_fails_before_device_probe(self, monkeypatch):
        import sjfpo.utils as utils

        monkeypatch.setenv("SJFPO_DTYPE", "fp16")
        with pytest.raises(ValueError, match="SJFPO_DTYPE"):
            utils.resolve_compute_dtype()


class TestMaybeCompile:
    def test_opt_out_returns_same_object(self, monkeypatch):
        import torch
        from sjfpo.utils import maybe_compile

        monkeypatch.setenv("SJFPO_COMPILE", "0")
        m = torch.nn.Linear(4, 4)
        assert maybe_compile(m) is m

    def test_cpu_device_is_not_compiled_even_when_cuda_is_available(
        self, monkeypatch
    ):
        import torch
        from sjfpo.utils import maybe_compile

        monkeypatch.delenv("SJFPO_COMPILE", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        m = torch.nn.Linear(4, 4)
        assert maybe_compile(m) is m
        assert maybe_compile(m, device=torch.device("cpu")) is m

    def test_auto_disables_on_cpu_without_cuda(self, monkeypatch):
        import torch
        from sjfpo.utils import maybe_compile

        monkeypatch.delenv("SJFPO_COMPILE", raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        m = torch.nn.Linear(4, 4)
        assert maybe_compile(m) is m  # CPU inductor guard bug -> eager
