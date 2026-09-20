"""Shared fixtures. Tests must run offline: nothing here downloads weights or datasets."""

import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# The reports only save figures. Without this, matplotlib picks the Tk backend on Windows/desktop Linux, and a
# broken or locked Tcl install then fails runs at random (TclError: Can't find a usable init.tcl).
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(scope="session")
def synthetic_folder(tmp_path_factory):
    """A tiny anomaly detection dataset on disk: textured normal images, test anomalies with masks."""
    root = tmp_path_factory.mktemp("synthetic")
    rng = np.random.default_rng(0)

    def image(anomalous: bool):
        base = cv2.GaussianBlur(rng.random((64, 64)).astype(np.float32), (0, 0), 6)
        base = 60 + 120 * (base - base.min()) / (np.ptp(base) + 1e-6)
        chip = (base * rng.gamma(20.0, 1 / 20.0, (64, 64))).clip(0, 255)
        mask = np.zeros((64, 64), np.uint8)
        if anomalous:
            chip[20:36, 20:36] = 250
            mask[20:36, 20:36] = 255
        return chip.astype(np.uint8), mask

    for directory in ("train/norm", "test/norm", "test/anom", "test/masks", "signatures"):
        (root / directory).mkdir(parents=True)
    for i in range(16):
        cv2.imwrite(str(root / f"train/norm/{i}.png"), image(False)[0])
    for i in range(6):
        cv2.imwrite(str(root / f"test/norm/{i}.png"), image(False)[0])
        chip, mask = image(True)
        cv2.imwrite(str(root / f"test/anom/{i}.png"), chip)
        cv2.imwrite(str(root / f"test/masks/{i}.png"), mask)
    for i in range(8):
        cv2.imwrite(str(root / f"signatures/{i}.png"), image(True)[0])
    return root


@pytest.fixture
def synthetic_datamodule(synthetic_folder):
    from anomalib.data import Folder

    def make(batch_size=8, num_workers=0, **kwargs):
        return Folder(
            name="synthetic", root=synthetic_folder, normal_dir="train/norm", abnormal_dir="test/anom",
            normal_test_dir="test/norm", mask_dir="test/masks", train_batch_size=batch_size,
            eval_batch_size=batch_size, num_workers=num_workers, **kwargs,
        )

    return make
