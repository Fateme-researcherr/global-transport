"""Tests for loss.py: couplings (independent / OT / class-conditional OT), the logvar loss,
REPA projection loss, guidance targets and the MeanFlow JVP target. CPU-only; the DMF path
runs on a tiny stub model with the DMFT signature, so flash-attn is not needed."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from loss import (
    DMFLoss,
    FMLoss,
    compute_guidance,
    interpolant,
    log_lv_loss,
    mse_loss,
    repa_loss,
)


# ===========================================================================
# log_lv_loss: per-sample logvar-weighted loss
# ===========================================================================

def test_log_lv_loss_is_per_sample_scalar():
    """log_lv_loss must produce a per-sample loss (shape (bs,)), not an outer product
    (bs,1,1,bs) that mixes logvar[i] with mean_loss[j]."""
    bs = 8
    x = torch.randn(bs, 4, 16, 16)
    y = torch.randn(bs, 4, 16, 16)
    lv = torch.zeros(bs, 1, 1, 1, requires_grad=True)  # DMFT.forward returns logvar as (bs,1,1,1)
    _, log_loss = log_lv_loss(x, y, lv)
    assert tuple(log_loss.shape) == (bs,), tuple(log_loss.shape)


def test_log_lv_loss_gradient_is_per_sample():
    """The gradient of log_loss wrt lv[i] must depend only on sample i's error."""
    bs = 4
    x = torch.zeros(bs, 2, 4, 4)
    y = torch.zeros(bs, 2, 4, 4)
    y[0] += 1.0  # only sample 0 has a nonzero error
    lv = torch.zeros(bs, 1, 1, 1, requires_grad=True)
    _, log_loss = log_lv_loss(x, y, lv)
    log_loss.sum().backward()
    g = lv.grad.reshape(-1)
    assert not torch.allclose(g[0], g[1])
    assert torch.allclose(g[1], g[2]) and torch.allclose(g[2], g[3])


# ===========================================================================
# repa_loss / mse_loss shape contracts used by FMLoss
# ===========================================================================

def test_repa_loss_returns_per_sample_or_scalar():
    bs, L, D = 32, 256, 768
    out = repa_loss([torch.randn(bs, L, D)], [torch.randn(bs, L, D)])
    assert out.dim() <= 1 and (out.shape[0] if out.dim() == 1 else 1) in (1, bs), tuple(out.shape)


def test_repa_loss_broadcasts_against_denoising_loss():
    bs, L, D = 64, 256, 768  # bs != L
    proj = repa_loss([torch.randn(bs, L, D)], [torch.randn(bs, L, D)])
    loss = torch.randn(bs) + 0.5 * proj  # must not raise
    assert loss.shape[0] == bs


def test_mse_loss_returns_per_sample_then_reduces():
    x = torch.randn(6, 4, 8, 8)
    y = torch.randn(6, 4, 8, 8)
    mse, loss = mse_loss(x, y)
    assert loss.shape == (6,)  # per-sample mean
    assert mse.shape == (6,)   # per-sample sum


# ===========================================================================
# interpolant: linear / cosine paths
# ===========================================================================

def test_interpolant_linear_path_endpoints():
    # x_t = (1-t)*x + t*eps, v_t = -x + eps  ->  x_t == (1-t)*x + t*(v_t + x) for any t.
    x = torch.randn(3, 4, 8, 8)
    x_t0, _ = interpolant(x, torch.zeros(3, 1, 1, 1), path_type="linear")
    assert torch.allclose(x_t0, x, atol=1e-6), "t=0 must give data x"
    for tval in (0.0, 0.25, 0.5, 0.75, 1.0):
        t = torch.full((3, 1, 1, 1), tval)
        x_t, v_t = interpolant(x, t, path_type="linear")
        assert torch.allclose(x_t, (1 - t) * x + t * (v_t + x), atol=1e-6), f"x_t identity fails at t={tval}"
    x_t1, _ = interpolant(x, torch.ones(3, 1, 1, 1), path_type="linear")
    assert not torch.allclose(x_t1, x, atol=1e-3), "t=1 must be noise, not data"


def test_interpolant_cosine_path_endpoints():
    x = torch.randn(3, 4, 8, 8)
    x_t0, _ = interpolant(x, torch.zeros(3, 1, 1, 1), path_type="cosine")
    assert torch.allclose(x_t0, x, atol=1e-6), "t=0 must give data x"
    x_t1, _ = interpolant(x, torch.ones(3, 1, 1, 1), path_type="cosine")
    assert not torch.allclose(x_t1, x, atol=1e-6), "t=1 must be noise, not data"


# ===========================================================================
# noise coupling inside interpolant: independent / OT / CCOT
# ===========================================================================

def _coupled_noise(x, **kwargs):
    """The noise interpolant draws for x: at t = 1 the linear path returns x_t = eps."""
    return interpolant(x, torch.ones(x.shape[0], 1, 1, 1), path_type="linear", **kwargs)[0]


def test_ot_coupling_aligns_noise_to_data():
    """With use_ot the returned noise is a permutation of the drawn noise that minimises
    the squared-Euclidean transport cost, so it must not cost more than the identity."""
    torch.manual_seed(0)
    x = torch.randn(32, 4, 8, 8)
    torch.manual_seed(1)
    eps_ind = _coupled_noise(x)                # independent draw
    torch.manual_seed(1)
    eps_ot = _coupled_noise(x, use_ot=True)    # same draw, OT-permuted
    assert torch.isfinite(eps_ot).all()
    assert sorted(eps_ot.reshape(32, -1).sum(1).tolist()) == pytest.approx(
        sorted(eps_ind.reshape(32, -1).sum(1).tolist()), abs=1e-4), "OT must permute rows, not change them"
    cost = lambda e: ((x - e) ** 2).sum().item()
    assert cost(eps_ot) <= cost(eps_ind) + 1e-4


def test_ccot_coupling_is_a_within_class_permutation():
    torch.manual_seed(0)
    x = torch.randn(16, 4, 8, 8)
    c = torch.arange(16) % 4
    torch.manual_seed(1)
    eps_ind = _coupled_noise(x)
    torch.manual_seed(1)
    eps_ccot = _coupled_noise(x, c=c, use_ccot=True)
    # every returned row is one of the drawn rows, and it came from a row of the same class
    flat_ind, flat_ccot = eps_ind.reshape(16, -1), eps_ccot.reshape(16, -1)
    for i in range(16):
        src = [k for k in range(16) if torch.allclose(flat_ccot[i], flat_ind[k])]
        assert len(src) == 1, "row is not a copy of exactly one drawn row"
        assert c[src[0]] == c[i], "noise was matched across classes"


# ===========================================================================
# compute_guidance: mg (DMF, 4-arg model) vs distill (SiT teacher, 3-arg model)
# ===========================================================================

def test_compute_guidance_mg_uses_dmf_4arg_signature():
    calls = []

    class RecDMF(nn.Module):
        def forward(self, x, t, r, y):
            calls.append(x.shape[0])
            return torch.randn_like(x)

    x = torch.randn(4, 2, 4, 4)
    t = torch.full((4, 1, 1, 1), 0.5)
    y = torch.randint(0, 10, (4,))
    v_t = torch.randn_like(x)
    out = compute_guidance(RecDMF(), x, t, y, v_t, torch.zeros(4, dtype=torch.bool),
                           omega=0.5, g_min=0.0, g_max=1.0, g_type="mg")
    assert calls == [8], "mg must evaluate cond + uncond in one doubled batch"
    assert out.shape == x.shape


def test_compute_guidance_mg_respects_interval_and_dropout():
    class ConstDMF(nn.Module):
        def forward(self, x, t, r, y):
            return torch.ones_like(x) * (y.reshape(-1, 1, 1, 1) == 1000).float()  # uncond -> 1, cond -> 0

    x = torch.randn(4, 2, 4, 4)
    y = torch.zeros(4, dtype=torch.long)
    v_t = torch.zeros_like(x)
    t = torch.tensor([0.1, 0.5, 0.9, 0.5]).reshape(4, 1, 1, 1)
    drop = torch.tensor([False, False, False, True])
    out = compute_guidance(ConstDMF(), x, t, y, v_t, drop, omega=0.5, g_min=0.3, g_max=0.7, g_type="mg")
    # v_cond - v_uncond = -1, so guided rows equal -omega; rows outside [g_min, g_max] or dropped stay v_t.
    assert torch.allclose(out[0], v_t[0]) and torch.allclose(out[2], v_t[2]) and torch.allclose(out[3], v_t[3])
    assert torch.allclose(out[1], torch.full_like(out[1], -0.5))


def test_compute_guidance_distill_uses_sit_3arg_signature():
    class RecSiT(nn.Module):
        def forward(self, x, t, y):
            return torch.randn_like(x)

    x = torch.randn(4, 2, 4, 4)
    t = torch.full((4, 1, 1, 1), 0.5)
    y = torch.randint(0, 10, (4,))
    out = compute_guidance(RecSiT(), x, t, y, torch.randn_like(x), drop_ids=torch.zeros(4, dtype=torch.bool),
                           omega=0.5, g_min=0.0, g_max=1.0, g_type="distill")
    assert out.shape == x.shape


# ===========================================================================
# DMFLoss / FMLoss end-to-end on tiny stub models
# ===========================================================================

class _StubDMF(nn.Module):
    """Minimal DMF-signature model: forward(x, t, r, y, return_logvar=False), logvar as (bs,1,1,1)."""

    def __init__(self, channels=4, size=8):
        super().__init__()
        self.in_channels = channels
        self.lin = nn.Linear(channels * size * size, channels * size * size)
        self.logvar = nn.Linear(1, 1)

    def forward(self, x, t, r, y, return_logvar=False):
        b = x.shape[0]
        out = self.lin(x.reshape(b, -1)).reshape(b, self.in_channels, *x.shape[2:])
        if return_logvar:
            return out, self.logvar(t.reshape(-1, 1)).reshape(b, 1, 1, 1)
        return out


class _StubSiT(nn.Module):
    """Minimal SiT-signature model: forward(x, t, y)."""

    def __init__(self, channels=4, size=8):
        super().__init__()
        self.in_channels = channels
        self.lin = nn.Linear(channels * size * size, channels * size * size)

    def forward(self, x, t, y):
        b = x.shape[0]
        return self.lin(x.reshape(b, -1)).reshape(b, self.in_channels, *x.shape[2:])


@pytest.mark.parametrize("kwargs", [dict(), dict(use_ot=True), dict(use_ccot=True)])
def test_dmf_loss_runs_for_every_coupling(kwargs):
    torch.manual_seed(0)
    model = _StubDMF()
    loss_fn = DMFLoss(P_mean=0.0, P_std=1.0, g_type="default", cfg_prob=0.0, **kwargs)
    x = torch.randn(4, 4, 8, 8)
    y = torch.randint(0, 10, (4,))
    loss, d = loss_fn(model, model, x, y)
    assert loss.shape == (4,)
    for k in ("fm_loss", "fm_p_loss", "mf_loss", "mf_p_loss"):
        assert d[k].shape == (4,), k
    loss.mean().backward()


@pytest.mark.parametrize("kwargs", [dict(), dict(use_ot=True), dict(use_ccot=True)])
def test_fm_loss_runs_for_every_coupling(kwargs):
    torch.manual_seed(0)
    model = _StubSiT()
    loss_fn = FMLoss(P_mean=0.0, P_std=1.0, cfg_prob=0.1, proj_coeff=0.0, **kwargs)
    x = torch.randn(4, 4, 8, 8)
    y = torch.randint(0, 10, (4,))
    loss, d = loss_fn(model, x, y)
    assert loss.shape == (4,) and d["fm_loss"].shape == (4,)
    loss.mean().backward()


def test_fm_loss_with_continuous_conditions():
    """Continuous conditions are float vectors; cfg dropout maps them to the zero vector."""
    torch.manual_seed(0)
    model = _StubSiT()
    loss_fn = FMLoss(P_mean=0.0, P_std=1.0, cfg_prob=0.1, proj_coeff=0.0, use_ot=True)
    x = torch.randn(4, 4, 8, 8)
    y = torch.randn(4, 16)
    loss, _ = loss_fn(model, x, y)
    assert loss.shape == (4,)
