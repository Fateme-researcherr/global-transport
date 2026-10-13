"""Multi-GPU coupling check: the OT and CCOT plans under DDP must equal the single-process plans.

Run with 2+ GPUs from the repository root:
    accelerate launch --multi_gpu --num_machines 1 --num_processes 2 tests/ddp_ot_equivalence.py
Every rank builds the same seeded 256-sample batch, calls the *_global samplers on its own
shard (real all-gather + broadcast), the outputs are gathered, and rank 0 compares them
row-by-row with the single-process samplers on the full batch. Prints RESULT: PASS/FAIL.
"""
import sys

import numpy as np
import torch
from accelerate import Accelerator

sys.path.insert(0, ".")
from optimal_transport import OTPlanSampler  # noqa: E402
from loss import interpolant  # noqa: E402

acc = Accelerator()
W, r, dev = acc.num_processes, acc.process_index, acc.device
G = 256
L = G // W
g = torch.Generator().manual_seed(0)  # identical global batch on every rank
x = torch.randn(G, 4, 32, 32, generator=g).to(dev)
eps = torch.randn(G, 4, 32, 32, generator=g).to(dev)
c = torch.randint(0, 10, (G,), generator=g).to(dev)
S = OTPlanSampler()
sl = slice(r * L, (r + 1) * L)  # this rank's shard

out_ot = acc.gather(S.sample_plan_with_scipy_global(x[sl], eps[sl], acc))             # rank order
out_ccot = acc.gather(S.sample_conditional_plan_with_scipy_global(x[sl], eps[sl], c[sl], acc))
ok_ot = ok_ccot = True
if acc.is_main_process:
    j_ot = S.sample_plan_with_scipy(x, eps)                     # single-process references
    j_ccot = S.sample_conditional_plan_with_scipy(x, eps, c)
    ok_ot = torch.equal(out_ot, eps[j_ot])
    ok_ccot = torch.equal(out_ccot, eps[j_ccot])
    j_loc = np.concatenate([S.sample_plan_with_scipy(x[k * L:(k + 1) * L], eps[k * L:(k + 1) * L]) + k * L
                            for k in range(W)])
    print(f"[{W} procs] global OT plan   == single-process assignment: {ok_ot}")
    print(f"[{W} procs] global CCOT plan == single-process assignment: {ok_ccot}")
    print(f"per-GPU {L}-plans differ from the global OT plan in {int((j_loc != j_ot).sum())}/{G} rows (expected: many)")

# the training entry point must take the gathered path and run
t1 = torch.ones(L, 1, 1, 1, device=dev)
assert interpolant(x[sl], t1, use_ot=True, accelerator=acc)[0].shape == (L, 4, 32, 32)
assert interpolant(x[sl], t1, c=c[sl], use_ccot=True, accelerator=acc)[0].shape == (L, 4, 32, 32)
acc.wait_for_everyone()
ok = ok_ot and ok_ccot
if acc.is_main_process:
    print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
