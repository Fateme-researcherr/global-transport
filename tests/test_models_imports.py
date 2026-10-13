"""The model modules must stay importable on machines without the flash-attn CUDA
extensions (CPU, CI), since attn_func="torch_sdpa"/"base" does not need them."""
import importlib
import os
import sys

import pytest


# --- models/layers.py must not hard-require the flash-attn CUDA extensions ---

def test_models_layers_importable_without_flash_attn():
    """models/layers.py does `from .flash_attention_2_jvp import ...` and
    `from .flash_attention_3_jvp import ...` at module top level, which import
    flash_attn_2_cuda / flash_attn_3._C. On any machine without those CUDA
    extensions (CPU, CI, non-Hopper/non-Ampere), importing models.layers (and
    therefore models.dmft / models.sit) raises ModuleNotFoundError — even when
    the user only wants attn_func='torch_sdpa'/'base'.

    A lazy import inside attn_op would fix this. This test asserts that
    importing models.layers succeeds without flash-attn installed."""
    # Make sure flash_attn is NOT importable in this env.
    try:
        import flash_attn_2_cuda  # noqa: F401
        pytest.skip("flash_attn_2_cuda is installed; cannot test the missing-ext path")
    except ModuleNotFoundError:
        pass
    # Importing must not fail due to the unconditional flash-attn imports.
    try:
        importlib.import_module("models.layers")
    except ModuleNotFoundError as e:
        if "flash_attn" in str(e) or "flash_attn_2_cuda" in str(e) or "flash_attn_3" in str(e):
            pytest.fail(
                "models.layers is un-importable without flash-attn installed "
                f"({e}). Move the flash-attn imports inside attn_op."
            )
        raise


def test_models_dmft_importable_without_flash_attn():
    try:
        import flash_attn_2_cuda  # noqa: F401
        pytest.skip("flash_attn installed")
    except ModuleNotFoundError:
        pass
    try:
        importlib.import_module("models.dmft")
    except ModuleNotFoundError as e:
        if "flash_attn" in str(e):
            pytest.fail(f"models.dmft un-importable without flash-attn ({e})")
        raise
