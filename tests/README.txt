Test suite for the repository (CPU-only, run from the repository root):

    python -m pytest tests/ -v

The loss / coupling / sampler tests use small stub models with the SiT and DMFT call
signatures, so flash-attn is not needed. tests/test_models_imports.py additionally checks
that the model modules import on a machine without the flash-attn CUDA extensions
(it is skipped where they are installed).

tests/ddp_ot_equivalence.py is a multi-GPU check that is NOT collected by pytest: it
verifies that the OT plan solved over the global batch under DDP equals the single-process
plan on the same batch.

    accelerate launch --multi_gpu --num_machines 1 --num_processes 2 tests/ddp_ot_equivalence.py
