
"""CPU multi-process regression test for global OT and CCOT."""
import os
import signal
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from optimal_transport import OTPlanSampler

WORLD_SIZE = 2
BATCH = 16
TIMEOUT = 240


def make_batch():
    g = torch.Generator().manual_seed(1234)
    x0 = torch.randn(BATCH, 2, 4, 4, generator=g)
    x1 = torch.randn(BATCH, 2, 4, 4, generator=g)
    c = torch.randint(0, 3, (BATCH,), generator=g)
    return x0, x1, c


def worker(outdir):
    from accelerate import Accelerator

    accelerator = Accelerator(cpu=True)
    try:
        assert accelerator.num_processes == WORLD_SIZE
        assert accelerator.device.type == "cpu"

        x0, x1, c = make_batch()
        b = BATCH // WORLD_SIZE
        sl = slice(
            accelerator.process_index * b,
            (accelerator.process_index + 1) * b,
        )
        sampler = OTPlanSampler()

        ot = sampler.sample_plan_with_scipy_global(
            x0[sl], x1[sl], accelerator
        )
        ccot = sampler.sample_conditional_plan_with_scipy_global(
            x0[sl], x1[sl], c[sl], accelerator
        )
        assert ot.shape == x1[sl].shape
        assert ccot.shape == x1[sl].shape

        torch.save(
            {"ot": ot, "ccot": ccot},
            os.path.join(outdir, f"rank{accelerator.process_index}.pt"),
        )
        accelerator.wait_for_everyone()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.fixture(scope="module")
def distributed_outputs(tmp_path_factory):
    pytest.importorskip("accelerate")
    if sys.platform == "win32":
        pytest.skip("This test uses POSIX process cleanup")

    outdir = str(tmp_path_factory.mktemp("ddp_ot"))
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", f"--nproc_per_node={WORLD_SIZE}",
        os.path.abspath(__file__), outdir,
    ]
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        log, _ = proc.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        log, _ = proc.communicate()
        pytest.fail(f"Workers timed out\n{log}")

    assert proc.returncode == 0, f"Worker failure:\n{log}"
    outputs = [
        torch.load(
            os.path.join(outdir, f"rank{r}.pt"),
            weights_only=True,
        )
        for r in range(WORLD_SIZE)
    ]
    return {
        key: torch.cat([item[key] for item in outputs], dim=0)
        for key in ("ot", "ccot")
    }


def recover_indices(output, x1):
    distances = torch.cdist(
        output.reshape(BATCH, -1), x1.reshape(BATCH, -1)
    )
    indices = distances.argmin(dim=1)
    assert torch.equal(x1[indices], output)
    assert torch.equal(indices.sort().values, torch.arange(BATCH))
    return indices


def cost(matrix, indices):
    return matrix.double()[torch.arange(BATCH), indices].sum().item()


def test_global_ot_matches_single_process(distributed_outputs):
    x0, x1, _ = make_batch()
    indices = recover_indices(distributed_outputs["ot"], x1)
    matrix = torch.cdist(x0.flatten(1), x1.flatten(1)) ** 2
    reference = torch.as_tensor(
        OTPlanSampler().sample_plan_with_scipy(x0, x1),
        dtype=torch.long,
    )
    assert cost(matrix, indices) == pytest.approx(
        cost(matrix, reference), rel=1e-5
    )
    assert torch.equal(indices, reference), (
        "OT assignment differs from single-process reference"
    )


def test_global_ccot_matches_single_process(distributed_outputs):
    x0, x1, c = make_batch()
    indices = recover_indices(distributed_outputs["ccot"], x1)
    matrix = OTPlanSampler.conditional_cost(x0, x1, c)
    reference = torch.as_tensor(
        OTPlanSampler().sample_conditional_plan_with_scipy(x0, x1, c),
        dtype=torch.long,
    )
    assert cost(matrix, indices) == pytest.approx(
        cost(matrix, reference), rel=1e-5
    )
    assert torch.equal(indices, reference), (
        "CCOT assignment differs from single-process reference"
    )


if __name__ == "__main__":
    args = [arg for arg in sys.argv[1:] if not arg.startswith("--")]
    worker(args[0])
