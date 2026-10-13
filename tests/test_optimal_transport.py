"""Tests for optimal_transport.py — OT coupling correctness and edge cases."""
import numpy as np
import torch

from optimal_transport import OTPlanSampler


# --- OT coupling direction -------------------------------------------------

def test_sample_plan_with_scipy_permutes_x1_not_x0():
    """sample_plan_with_scipy must return the column indices j that align x1
    (noise) to x0 (data) row-by-row. Permuting x0 would invert the coupling."""
    sampler = OTPlanSampler()
    # data: two clusters far apart; noise: opposite clusters so OT must swap.
    x0 = torch.tensor([[0.0], [0.0], [10.0], [10.0]])
    x1 = torch.tensor([[10.0], [0.0], [10.0], [0.0]])
    j = sampler.sample_plan_with_scipy(x0, x1)
    j = np.asarray(j)
    # x0 unchanged; x1[j] should now be close to x0 row-by-row.
    aligned = x1[torch.as_tensor(j, dtype=torch.long)]
    assert torch.allclose(aligned, x0), (aligned.tolist(), x0.tolist())


# --- Class-conditional OT (CCOT) --------------------------------------------

def _ccot_batch():
    torch.manual_seed(0)
    x0 = torch.randn(16, 3)
    x1 = torch.randn(16, 3)
    c = torch.arange(16) % 4
    return x0, x1, c


def test_conditional_plan_is_a_within_class_permutation():
    sampler = OTPlanSampler()
    x0, x1, c = _ccot_batch()
    j = torch.as_tensor(sampler.sample_conditional_plan_with_scipy(x0, x1, c), dtype=torch.long)
    assert sorted(j.tolist()) == list(range(16)), "not a permutation"
    # noise k carries the label of datapoint k, so matched rows must share the class
    assert (c[j] == c).all(), "some datapoint was coupled to noise of another class"


def test_conditional_plan_equals_ot_within_each_class():
    """With the default barrier the plan is exactly the unconditional OT plan restricted
    to every class."""
    sampler = OTPlanSampler()
    x0, x1, c = _ccot_batch()
    j = torch.as_tensor(sampler.sample_conditional_plan_with_scipy(x0, x1, c), dtype=torch.long)
    for k in range(4):
        idx = (c == k).nonzero().flatten()
        j_k = torch.as_tensor(sampler.sample_plan_with_scipy(x0[idx], x1[idx]), dtype=torch.long)
        assert torch.equal(j[idx], idx[j_k]), f"class {k}: within-class plan differs from OT"


def test_conditional_plan_with_zero_barrier_is_plain_ot():
    sampler = OTPlanSampler()
    x0, x1, c = _ccot_batch()
    j_ccot = sampler.sample_conditional_plan_with_scipy(x0, x1, c, beta_mult=0.0)
    j_ot = sampler.sample_plan_with_scipy(x0, x1)
    assert np.array_equal(np.asarray(j_ccot), np.asarray(j_ot))
