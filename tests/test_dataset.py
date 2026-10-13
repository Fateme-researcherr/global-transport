"""Tests for dataset.py — label loading robustness."""
import json
import os

import numpy as np
import pytest
import torch

import dataset as ds


def _write_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def _make_features_dir(root, fnames, labels_map):
    feats = os.path.join(root, "vae-sd")
    os.makedirs(feats, exist_ok=True)
    for f in fnames:
        _write_npy(os.path.join(feats, f), np.random.randn(8, 8, 8).astype(np.float32))
    with open(os.path.join(feats, "dataset.json"), "w") as f:
        json.dump({"labels": labels_map}, f)


def _make_images_dir(root, fnames, labels_map):
    imgs = os.path.join(root, "images")
    os.makedirs(imgs, exist_ok=True)
    from PIL import Image
    for f in fnames:
        d = os.path.dirname(os.path.join(imgs, f))
        os.makedirs(d, exist_ok=True)
        Image.fromarray((np.random.rand(8, 8, 3) * 255).astype(np.uint8)).save(
            os.path.join(imgs, f))
    with open(os.path.join(imgs, "dataset.json"), "w") as f:
        json.dump({"labels": labels_map}, f)


# --- dataset.json out of sync with the files on disk -------------------

def test_vae_dataset_missing_fname_gives_clear_error(tmp_path):
    fnames = ["00000/img-mean-std-00000000.npy"]
    # labels dict missing the on-disk fname
    _make_features_dir(str(tmp_path), fnames, labels_map={})
    with pytest.raises(KeyError) as excinfo:
        ds.VAELabelDataset(str(tmp_path))
    # The error message should mention the missing fname (not just a bare key).
    assert "img-mean-std-00000000" in str(excinfo.value) or "missing" in str(excinfo.value).lower(), (
        f"Uninformative KeyError: {excinfo.value}"
    )


# --- Happy path: scalar int labels -> int64 ------------------------------

def test_vae_dataset_loads_int_labels(tmp_path):
    fnames = ["00000/a.npy", "00000/b.npy"]
    labels = {fnames[0]: 5, fnames[1]: 7}
    _make_features_dir(str(tmp_path), fnames, labels_map=labels)
    d = ds.VAELabelDataset(str(tmp_path))
    assert d.labels.dtype == np.int64
    feat, lbl = d[0]
    assert isinstance(feat, torch.Tensor) and isinstance(lbl, torch.Tensor)
