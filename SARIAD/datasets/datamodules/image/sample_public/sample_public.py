"""SAMPLE: Synthetic and Measured Paired Labeled Experiment (public release).

10 vehicle classes as 128x128 SAR chips, each available as *measured* (``real``, MSTAR) and as
CAD-simulated (``synth``) images, with two amplitude scalings (``qpm``: quarter-power magnitude,
``decibel``). https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public (about 1.5 GB download).

The dataset has no anomaly-detection labels. Like MSTAR, it is turned into a benchmark by
treating the vehicle as the anomaly: the target and its shadow are segmented and removed to get
a *normal* image (:func:`SARIAD.utils.normal_gen.remove_target`), the original chip is the
anomalous image and the removed area is its mask.

Train and test are split by **azimuth** (test: the last ``test_azimuth_ratio`` of the 360 degrees), so
that neighbouring aspect angles of one vehicle, which look almost identical, never end up on both sides.
"""

import logging
import os
import re
from pathlib import Path

import cv2
import numpy as np
from anomalib.data import Folder

from SARIAD.config import DATASETS_PATH, DEBUG
from SARIAD.utils.blob_utils import fetch_blob
from SARIAD.utils.normal_gen import SegmentationFailed, remove_target

logger = logging.getLogger(__name__)

NAME = "SAMPLE_dataset_public"
LINK = "https://github.com/benjaminlewis-afrl/SAMPLE_dataset_public/archive/refs/heads/master.zip"

DOMAINS = ("real", "synth")
SCALINGS = ("qpm", "decibel")
_AZIMUTH = re.compile(r"azCenter_(\d+)")


def generate_sample(
    source: str | Path,
    output: str | Path,
    domain: str = "real",
    scaling: str = "qpm",
    test_azimuth_ratio: float = 0.2,
    segmenter: str = "classical",
    seed: int = 0,
) -> dict[str, int]:
    """Create ``{train,test}/{anom,norm,masks}`` from the SAMPLE PNG images.

    Chips whose target cannot be segmented reliably (see :class:`SegmentationFailed`) are skipped
    rather than written as a "normal" image that still contains the target.

    Returns:
        Number of chips ``kept`` and ``skipped``.
    """
    images = sorted(Path(source, "png_images", scaling, domain).glob("*/*.png"))
    if not images:
        msg = f"No SAMPLE images found in {Path(source, 'png_images', scaling, domain)}"
        raise FileNotFoundError(msg)

    for split in ("train", "test"):
        for kind in ("anom", "norm", "masks"):
            Path(output, split, kind).mkdir(parents=True, exist_ok=True)

    threshold = 360.0 * (1.0 - test_azimuth_ratio)
    counts = {"kept": 0, "skipped": 0, "train": 0, "test": 0}
    for index, path in enumerate(images):
        azimuth = _AZIMUTH.search(path.name)
        split = "test" if azimuth and float(azimuth.group(1)) >= threshold else "train"
        chip = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if chip is None:
            counts["skipped"] += 1
            continue
        try:
            normal, mask = remove_target(chip.astype(np.float32)[..., None] / 255.0, segmenter=segmenter, rng=np.random.default_rng([seed, index]))
        except SegmentationFailed as e:
            logger.debug("Skipping %s: %s", path.name, e)
            counts["skipped"] += 1
            continue
        name = f"{path.parent.name}_{path.name}"
        cv2.imwrite(str(Path(output, split, "anom", name)), chip)
        cv2.imwrite(str(Path(output, split, "norm", name)), np.clip(normal[..., 0] * 255, 0, 255).astype(np.uint8))
        cv2.imwrite(str(Path(output, split, "masks", name)), mask * 255)
        counts["kept"] += 1
        counts[split] += 1

    if counts["skipped"]:
        logger.warning("SAMPLE (%s, %s): %d of %d chips skipped (target segmentation failed)", domain, scaling, counts["skipped"], len(images))
    if counts["train"] == 0 or counts["test"] == 0:
        msg = f"SAMPLE split is empty ({counts}); check test_azimuth_ratio={test_azimuth_ratio} and the segmentation"
        raise RuntimeError(msg)
    return counts


class SAMPLE_PUBLIC(Folder):
    """SAMPLE anomaly detection datamodule (vehicle = anomaly).

    Args:
        domain: ``"real"`` (measured) or ``"synth"`` (simulated).
        scaling: ``"qpm"`` (default; the target segmentation assumes a dark clutter background) or
            ``"decibel"``, where many chips cannot be segmented and are skipped.
        test_azimuth_ratio: Fraction of azimuth angles (the highest ones) used for testing.
        segmenter: Backend that finds the target, see :mod:`SARIAD.utils.normal_gen`.
        batch_size: Train/eval batch size (1 when DEBUG is set).
        num_workers: Dataloader workers.
        path: Dataset directory (default ``<DATASETS_PATH>/SAMPLE_dataset_public``).
        seed: Seed of the normal-image generation.
        **folder_kwargs: Any other ``anomalib.data.Folder`` argument.
    """

    def __init__(
        self,
        domain: str = "real",
        scaling: str = "qpm",
        test_azimuth_ratio: float = 0.2,
        segmenter: str = "classical",
        batch_size: int = 16,
        num_workers: int = 8,
        path: str | None = None,
        seed: int = 0,
        **folder_kwargs,
    ) -> None:
        if domain not in DOMAINS or scaling not in SCALINGS:
            msg = f"domain must be one of {DOMAINS} and scaling one of {SCALINGS}, got {domain!r}, {scaling!r}"
            raise ValueError(msg)
        self.dataset_root = path or os.path.join(DATASETS_PATH, NAME)
        self.image_size = (128, 128)

        fetch_blob(self.dataset_root, link=LINK)
        benchmark_root = os.path.join(self.dataset_root, "anomaly_detection", f"{domain}_{scaling}")
        if not os.path.exists(os.path.join(benchmark_root, "test", "norm")):
            generate_sample(self.dataset_root, benchmark_root, domain, scaling, test_azimuth_ratio, segmenter, seed)

        batch_size = 1 if DEBUG else batch_size
        super().__init__(
            name=f"{NAME}_{domain}_{scaling}",
            root=benchmark_root,
            normal_dir="train/norm",
            abnormal_dir="test/anom",
            normal_test_dir="test/norm",
            mask_dir="test/masks",
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=num_workers,
            **folder_kwargs,
        )
