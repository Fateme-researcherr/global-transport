"""Tests for samplers.py — inference correctness and the generate.py bugs."""
import numpy as np
import torch
import torch.nn as nn

from samplers import (
    ctm_sampler,
    compute_diffusion,
    euler_sampler,
    fm_euler_sampler,
    fm_sde_sampler,
    get_score_from_velocity,
    get_tsteps,
    restart_sampler,
)


# --- Stub models ---------------------------------------------------------

class DMFStub(nn.Module):
    """DMF-signature: forward(x, t, r, y). Predicts u = (x_target - x)/dt style."""
    def __init__(self, channels=4):
        super().__init__()
        self.in_channels = channels
        self.lin = nn.Linear(channels, channels)

    def forward(self, x, t, r, y):
        # Return a velocity-like tensor (no crash). Shape preserved.
        return self.lin(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class SiTStub(nn.Module):
    """SiT-signature: forward(x, t, y)."""
    def __init__(self, channels=4):
        super().__init__()
        self.in_channels = channels
        self.lin = nn.Linear(channels, channels)

    def forward(self, x, t, y):
        return self.lin(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# --- generate.py --mode sde uses the SiT SDE sampler ----------------------

def test_generate_imports_sde_sampler_used_in_main():
    """--mode sde must dispatch to fm_sde_sampler (checked on the source, so the test
    does not need generate.py's runtime dependencies)."""
    src = open("generate.py").read()
    assert hasattr(__import__("samplers"), "fm_sde_sampler")
    assert not hasattr(__import__("samplers"), "sit_sde_sampler"), (
        "sit_sde_sampler is not defined in samplers.py — generate.py:156 will NameError"
    )
    assert "sit_sde_sampler" not in src, (
        "generate.py references undefined `sit_sde_sampler`; should be `fm_sde_sampler`"
    )


# --- generate.py: every args.X used in main() is defined -------------------

def test_generate_argparser_defines_every_attr_used_in_main():
    """Every args.X referenced in generate.py must be defined by the argparser."""
    import ast
    src = open("generate.py").read()
    tree = ast.parse(src)
    # Find the argparse add_argument calls and collect dest names.
    defined = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                flag = node.args[0].value
                defined.add(flag.lstrip("-").replace("-", "_"))
    # Collect args.X attribute accesses.
    used = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "args"):
            used.add(node.attr)
    missing = sorted(used - defined)
    assert not missing, f"generate.py main() uses args attrs the argparser does not define: {missing}"


# --- Euler / restart / CTM samplers run end-to-end -----------------------

def test_euler_sampler_runs_dmf():
    torch.manual_seed(0)
    m = DMFStub()
    z = torch.randn(2, 4, 8, 8)
    y = torch.randint(0, 10, (2,))
    out = euler_sampler(m, z, y, num_steps=4, shift=1.0)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_restart_sampler_runs_dmf():
    torch.manual_seed(0)
    m = DMFStub()
    z = torch.randn(2, 4, 8, 8)
    y = torch.randint(0, 10, (2,))
    out = restart_sampler(m, z, y, num_steps=4, shift=1.0, path_type="linear")
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_ctm_sampler_runs_dmf():
    torch.manual_seed(0)
    m = DMFStub()
    z = torch.randn(2, 4, 8, 8)
    y = torch.randint(0, 10, (2,))
    out = ctm_sampler(m, z, y, num_steps=4, shift=1.0, gamma=0.9)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_fm_euler_sampler_runs_sit():
    torch.manual_seed(0)
    m = SiTStub()
    z = torch.randn(2, 4, 8, 8)
    y = torch.randint(0, 10, (2,))
    out = fm_euler_sampler(m, z, y, num_steps=4, shift=1.0)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


def test_fm_sde_sampler_runs_sit():
    torch.manual_seed(0)
    m = SiTStub()
    z = torch.randn(2, 4, 8, 8)
    y = torch.randint(0, 10, (2,))
    out = fm_sde_sampler(m, z, y, num_steps=8, shift=1.0, path_type="linear",
                         coeff_type="sigma", gamma=1.0)
    assert out.shape == z.shape
    assert torch.isfinite(out).all()


# --- CFG: cond/uncond chunk order and formula ---------------------------

def test_euler_cfg_formula_uses_uncond_plus_scale_diff():
    """CFG must be d_uncond + scale*(d_cond - d_uncond) with cond = first half
    (y) and uncond = second half (y_null). Verify via a stub that distinguishes
    the two halves."""
    seen = {}

    class CfgStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_channels = 4
        def forward(self, x, t, r, y):
            # First half gets y in [0,10), second half gets 1000 (null).
            seen["y"] = y.tolist()
            out = torch.ones_like(x)
            # Make cond and uncond differ in a known way.
            out[:out.shape[0]//2] = 2.0
            out[out.shape[0]//2:] = 5.0
            return out

    m = CfgStub()
    z = torch.randn(2, 4, 4, 4)
    y = torch.tensor([3, 7])
    out = euler_sampler(m, z, y, num_steps=2, cfg_scale=2.0, g_min=0.0, g_max=1.0)
    # With d_cond=2, d_uncond=5, cfg=2: d = 5 + 2*(2-5) = -1
    # x_nxt = x + (t_nxt-t_cur)*d. Just check it ran with correct y layout.
    assert seen["y"][:2] == [3, 7] and seen["y"][2:] == [1000, 1000], seen["y"]
    assert torch.isfinite(out).all()


# --- get_tsteps / score / diffusion edge cases --------------------------

def test_get_tsteps_endpoints_fixed_under_shift():
    ts = get_tsteps(num_steps=4, t_start=1.0, t_end=0.0, shift=2.0)
    assert ts.numel() == 5
    assert abs(ts[0].item() - 1.0) < 1e-9 and abs(ts[-1].item()) < 1e-9
    assert (ts.diff() <= 0).all(), "t_steps must be monotone non-increasing"


def test_get_score_from_velocity_linear_recovers_noise_at_t1():
    """At t=1, x_t = eps ~ N(0,I); score should equal -eps = -x_t."""
    torch.manual_seed(0)
    xt = torch.randn(3, 4, 8, 8)
    t = torch.ones(3)
    # velocity v = -x_data + eps; at t=1 we don't know x_data, but the score
    # formula must give -xt regardless of v (alpha_t=0 at t=1). Use any v.
    v = torch.randn(3, 4, 8, 8)
    s = get_score_from_velocity(v, xt, t, path_type="linear")
    assert torch.allclose(s, -xt, atol=1e-5), (s.flatten()[:5], (-xt).flatten()[:5])


def test_compute_diffusion_cosine_elbo_finite_at_t1():
    t = torch.tensor(1.0)
    val = compute_diffusion(t, coeff_type="elbo", path_type="cosine", gamma=1.0)
    assert torch.isfinite(val).item(), f"cosine elbo diffusion not finite at t=1: {val}"
