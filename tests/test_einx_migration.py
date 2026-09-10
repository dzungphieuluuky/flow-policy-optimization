"""einx migration — torch↔einx numerical equivalence at 1e-6 (fp32).

Fervers et al. ICLR 2026 Oral arXiv:2607.27987 — ground truth is current torch behavior;
einx must match torch at atol=1e-6. Characterization + migration guard.
"""
import torch
import einx
import pytest


def assert_einx_equals_torch(fn_torch, fn_einx, atol=1e-6):
    y_torch, y_einx = fn_torch(), fn_einx()
    # support tuple outputs (e.g., split)
    if isinstance(y_torch, tuple):
        assert isinstance(y_einx, tuple)
        for a, b in zip(y_torch, y_einx):
            assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
            assert torch.allclose(a.float(), b.float(), atol=atol), f"max diff {(a-b).abs().max()}"
    else:
        assert y_torch.shape == y_einx.shape, f"shape mismatch {y_torch.shape} vs {y_einx.shape}"
        assert torch.allclose(y_torch.float(), y_einx.float(), atol=atol), f"max diff {(y_torch-y_einx).abs().max()}"


class TestEinxGuard:
    def test_import(self):
        import einx as _einx
        assert _einx.__version__.startswith("0.4")

    def test_graph_emits_torch(self):
        x = torch.randn(2, 3, 4)
        code = einx.id("b n d -> b (n d)", x, graph=True)
        assert "torch.reshape" in code or "reshape" in code

    def test_id_reshape(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 4)
        assert_einx_equals_torch(
            lambda: x.reshape(2, 12),
            lambda: einx.id("b n d -> b (n d)", x),
        )

    def test_unsqueeze_squeeze(self):
        torch.manual_seed(0)
        x = torch.randn(8)
        assert_einx_equals_torch(
            lambda: x.unsqueeze(0),
            lambda: einx.id("d -> 1 d", x),
        )
        y = torch.randn(1, 8, 7)
        assert_einx_equals_torch(
            lambda: y.squeeze(0),
            lambda: einx.id("1 h a -> h a", y),
        )

    def test_expand_broadcast(self):
        torch.manual_seed(0)
        x = torch.randn(4, 16, 128)
        h = torch.hann_window(16)
        assert_einx_equals_torch(
            lambda: x * h.view(1, -1, 1),
            lambda: einx.multiply("b n d, n -> b n d", x, h),
        )

    def test_cat_concat(self):
        torch.manual_seed(0)
        a = torch.randn(2, 8, 7)
        b = torch.randn(2, 8, 128)
        assert_einx_equals_torch(
            lambda: torch.cat([a, b], dim=-1),
            lambda: einx.id("b h da, b h db -> b h (da + db)", a, b),
        )

    def test_view_singleton(self):
        torch.manual_seed(0)
        t = torch.randn(4)
        assert_einx_equals_torch(
            lambda: t.view(4, 1, 1),
            lambda: einx.id("b -> b 1 1", t),
        )

    def test_dot(self):
        torch.manual_seed(0)
        flat = torch.randn(64, 128)
        W = torch.randn(128, 32)
        assert_einx_equals_torch(
            lambda: flat @ W,
            lambda: einx.dot("n d, d p -> n p", flat, W),
        )


class TestSpectralTokensEinx:
    """SpectralTokens hann + rFFT cat equivalence (networks.py:262-298)."""

    def test_hann_multiply(self):
        torch.manual_seed(0)
        for B, N, D in [(2, 16, 128), (4, 16, 64)]:
            z = torch.randn(B, N, D)
            hann = torch.hann_window(N)
            y_torch = z * hann.view(1, -1, 1)
            y_einx = einx.multiply("b n d, n -> b n d", z, hann)
            assert torch.allclose(y_torch, y_einx, atol=1e-6)

    def test_cat_real_imag(self):
        torch.manual_seed(0)
        real = torch.randn(2, 9, 32)
        imag = torch.randn(2, 9, 32)
        y_torch = torch.cat([real, imag], dim=-1)
        y_einx = einx.id("b k d, b k d -> b k (d + d)", real, imag)
        assert torch.allclose(y_torch, y_einx, atol=1e-6)

    def test_reshape_flat(self):
        torch.manual_seed(0)
        feat = torch.randn(2, 9, 64)
        D2 = 32
        y_torch = feat.reshape(-1, 64)
        y_einx = einx.id("b k d -> (b k) d", feat)
        assert torch.allclose(y_torch, y_einx, atol=1e-6)


class TestViTSIGRegEinx:
    def test_reshape_b_flat(self):
        torch.manual_seed(0)
        z = torch.randn(2, 16, 128)
        assert torch.allclose(z.reshape(2, -1), einx.id("b n d -> b (n d)", z), atol=1e-6)

    def test_dot_sigreg(self):
        torch.manual_seed(0)
        flat = torch.randn(32, 128)
        W = torch.randn(128, 32)
        assert torch.allclose(flat @ W, einx.dot("n d, d p -> n p", flat, W), atol=1e-6)

    def test_sigreg_broadcast(self):
        torch.manual_seed(0)
        proj = torch.randn(32, 32)
        t_grid = torch.linspace(0, 3, 10)
        y_torch = proj.unsqueeze(-1) * t_grid.unsqueeze(0).unsqueeze(0)
        y_einx = einx.multiply("n p, f -> n p f", proj, t_grid)
        assert torch.allclose(y_torch, y_einx, atol=1e-6)

    def test_vit_reshape(self):
        torch.manual_seed(0)
        x = torch.randn(2, 16, 3, 256, 256)
        B, T = 2, 16
        y_torch = x.reshape(B * T, 3, 256, 256)
        y_einx = einx.id("b t c h w -> (b t) c h w", x)
        assert torch.allclose(y_torch, y_einx, atol=1e-6)


class TestFusionJFTEinx:
    def test_temb_broadcast(self):
        torch.manual_seed(0)
        temb = torch.randn(4, 8)
        H = 8
        # old: temb.unsqueeze(1).expand(-1, H, -1)
        y_torch = temb.unsqueeze(1).expand(-1, H, -1)
        # einx: broadcast via id with explicit axis
        y_einx = einx.id("b d -> b h d", temb, h=H)
        assert torch.allclose(y_torch, y_einx, atol=1e-6)

    def test_cat_temb(self):
        torch.manual_seed(0)
        x_t = torch.randn(2, 8, 7)
        temb = torch.randn(2, 8)
        H = 8
        y_torch = torch.cat([x_t, temb.unsqueeze(1).expand(-1, H, -1)], dim=-1)
        temb_b = einx.id("b d -> b h d", temb, h=H)
        y_einx = einx.id("b h da, b h db -> b h (da + db)", x_t, temb_b)
        assert torch.allclose(y_torch, y_einx, atol=1e-6)


class TestDCTTrainerEinx:
    def test_arange_unsqueeze(self):
        n = 8
        arange = torch.arange(n, dtype=torch.float32)
        assert torch.allclose(arange.unsqueeze(1), einx.id("n -> n 1", arange), atol=1e-6)
        assert torch.allclose(arange.unsqueeze(0), einx.id("n -> 1 n", arange), atol=1e-6)

    def test_cat_joint(self):
        torch.manual_seed(0)
        A_norm = torch.randn(2, 8, 7)
        S_norm = torch.randn(2, 8, 128)
        y_torch = torch.cat([A_norm, S_norm], dim=-1)
        y_einx = einx.id("b h da, b h ds -> b h (da + ds)", A_norm, S_norm)
        assert torch.allclose(y_torch, y_einx, atol=1e-6)

    def test_trainer_view(self):
        torch.manual_seed(0)
        t = torch.randn(4)
        assert torch.allclose(t.view(4, 1, 1), einx.id("b -> b 1 1", t), atol=1e-6)

    def test_trainer_reshape_nm(self):
        torch.manual_seed(0)
        x = torch.randn(4, 3, 8, 7)
        assert torch.allclose(
            x.reshape(12, 8, 7), einx.id("n m h a -> (n m) h a", x), atol=1e-6
        )


class TestFPOEinx:
    def test_expand(self):
        torch.manual_seed(0)
        t = torch.randn(5)
        b, n_mc = 4, 3
        # t of shape [5] -> expand to [4,5] not relevant; test scalar expand pattern from fpo_core
        # fpo_core: t.expand(b, n_mc) where t is [n_mc]
        t2 = torch.randn(3)
        assert torch.allclose(t2.expand(4, 3), einx.id("m -> b m", t2, b=4), atol=1e-6)

    def test_reshape_flat(self):
        torch.manual_seed(0)
        t = torch.randn(4, 3)
        assert torch.allclose(t.reshape(-1), einx.id("b m -> (b m)", t), atol=1e-6)

    def test_repeat_interleave_uniform(self):
        torch.manual_seed(0)
        a = torch.randn(4, 7)
        y_torch = a.repeat_interleave(3, dim=0)
        y_einx = einx.id("b d -> (b m) d", a, m=3)
        assert torch.allclose(y_torch, y_einx, atol=1e-6)


class TestAdaptersEinx:
    def test_obs_unsqueeze(self):
        obs = torch.randn(8)
        assert torch.allclose(obs.unsqueeze(0), einx.id("d -> 1 d", obs), atol=1e-6)

    def test_squeeze(self):
        x = torch.randn(1, 8, 7)
        assert torch.allclose(x.squeeze(0), einx.id("1 h a -> h a", x), atol=1e-6)

    def test_context_expand(self):
        ctx = torch.randn(8)
        # runner.py: context.unsqueeze(0).expand(cfg.context_len, -1) where context_len=16
        y_torch = ctx.unsqueeze(0).expand(16, -1)
        y_einx = einx.id("d -> t d", ctx, t=16)
        assert torch.allclose(y_torch, y_einx, atol=1e-6)
