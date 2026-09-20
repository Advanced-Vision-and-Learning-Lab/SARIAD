"""SARDet-100K: large-scale multi-class SAR object detection dataset (NeurIPS 2024).

https://arxiv.org/abs/2403.06534, Kaggle: ``greatbird/sardet-100k`` (tens of GB; needs Kaggle
credentials). Layout: ``Annotations/{train,val,test}.json`` (COCO) and ``JPEGImages/``.

The dataset only has bounding boxes and no anomaly labels. It is turned into an anomaly-detection
benchmark by treating the annotated objects (ships, aircraft, vehicles, ...) as anomalies: the
original image is the anomalous one, its mask is the union of the (dilated) boxes, and the normal image is
the original with those boxes filled with clutter from the rest of the image
(:func:`SARIAD.utils.normal_gen.generate_normal`). Masks are therefore box-shaped.

Images are processed lazily and only up to ``max_images`` per split, because generating normal images is
slow for large images (patch inpainting).
"""

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from anomalib.data import Folder

from SARIAD.config import DATASETS_PATH, DEBUG
from SARIAD.utils.blob_utils import fetch_blob
from SARIAD.utils.normal_gen import generate_normal

logger = logging.getLogger(__name__)

NAME = "SARDet_100K"
KAGGLE = "greatbird/sardet-100k"


def find_root(path: str | Path) -> Path:
    """Locate the directory that contains ``Annotations/`` (the Kaggle archive may add nesting)."""
    path = Path(path)
    for candidate in [path, *sorted(p for p in path.glob("*") if p.is_dir()), *sorted(path.glob("*/*/"))]:
        if (candidate / "Annotations").is_dir():
            return candidate
    msg = f"No 'Annotations' directory found in {path}: is this a SARDet-100K download?"
    raise FileNotFoundError(msg)


def _image_path(root: Path, split: str, file_name: str) -> Path | None:
    for candidate in (root / "JPEGImages" / file_name, root / "JPEGImages" / split / file_name, root / file_name):
        if candidate.is_file():
            return candidate
    return None


def generate_sardet(
    root: str | Path,
    output: str | Path,
    max_images: int = 1000,
    categories: list[str] | None = None,
    min_box: int = 8,
    max_mask_fraction: float = 0.25,
    dilate: int = 15,
    fill: str = "patch",
    seed: int = 0,
) -> dict[str, int]:
    """Create ``{train,test}/{anom,norm,masks}`` from the COCO annotations of SARDet-100K.

    The ``train`` split comes from ``train.json`` and the ``test`` split from ``test.json``
    (``val.json`` if there is none). Images without a usable box are skipped, as are images where
    the boxes cover more than ``max_mask_fraction`` (little clutter left to fill from).

    Returns:
        Number of images written per split.
    """
    root = Path(root)
    written = {"train": 0, "test": 0}
    for split, annotation in (("train", "train"), ("test", "test")):
        annotation_file = root / "Annotations" / f"{annotation}.json"
        if not annotation_file.is_file() and split == "test":
            annotation_file = root / "Annotations" / "val.json"
        coco = json.loads(annotation_file.read_text())

        wanted = None
        if categories:
            wanted = {c["id"] for c in coco["categories"] if c["name"] in set(categories)}
            if not wanted:
                msg = f"None of {categories} in the dataset categories: {[c['name'] for c in coco['categories']]}"
                raise ValueError(msg)
        boxes: dict[int, list[list[float]]] = {}
        for ann in coco["annotations"]:
            if wanted is not None and ann["category_id"] not in wanted:
                continue
            if min(ann["bbox"][2:]) >= min_box:
                boxes.setdefault(ann["image_id"], []).append(ann["bbox"])

        for kind in ("anom", "norm", "masks"):
            Path(output, split, kind).mkdir(parents=True, exist_ok=True)

        images = sorted((im for im in coco["images"] if im["id"] in boxes), key=lambda im: im["id"])
        rng = np.random.default_rng([seed, 0 if split == "train" else 1])
        for image_info in (images[i] for i in rng.permutation(len(images))):
            if written[split] >= max_images:
                break
            path = _image_path(root, split, image_info["file_name"])
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path else None
            if image is None:
                continue
            mask = np.zeros(image.shape, np.uint8)
            for x, y, w, h in boxes[image_info["id"]]:
                mask[max(int(y), 0) : int(np.ceil(y + h)), max(int(x), 0) : int(np.ceil(x + w))] = 1
            if mask.mean() > max_mask_fraction:
                continue
            normal = generate_normal(image, mask, method=fill, dilate=dilate, rng=rng)
            name = f"{image_info['id']}.png"
            cv2.imwrite(str(Path(output, split, "anom", name)), image)
            cv2.imwrite(str(Path(output, split, "norm", name)), normal)
            cv2.imwrite(str(Path(output, split, "masks", name)), mask * 255)
            written[split] += 1

    if not all(written.values()):
        msg = f"SARDet-100K generation produced no images for a split ({written})"
        raise RuntimeError(msg)
    logger.info("SARDet-100K anomaly benchmark generated: %s", written)
    return written


class SARDet_100K(Folder):
    """SARDet-100K anomaly detection datamodule (annotated objects = anomalies).

    Args:
        max_images: Images generated per split (normal-image generation is slow). Defaults to 1000.
        categories: Only use objects of these COCO categories (e.g. ``["ship"]``); ``None`` uses all.
        min_box: Ignore boxes with a side shorter than this many pixels.
        max_mask_fraction: Skip images whose boxes cover more than this fraction.
        dilate: Box dilation (pixels) before filling, to cover the smeared edges of the objects.
        fill: ``"patch"`` (better clutter, slow) or ``"background"`` (fast, i.i.d. pixels).
        batch_size: Train/eval batch size (1 when DEBUG is set).
        num_workers: Dataloader workers.
        path: Dataset directory (default ``<DATASETS_PATH>/SARDet_100K``); downloaded from Kaggle if empty.
        seed: Seed for the image subset and the normal-image generation.
        **folder_kwargs: Any other ``anomalib.data.Folder`` argument.
    """

    def __init__(
        self,
        max_images: int = 1000,
        categories: list[str] | None = None,
        min_box: int = 8,
        max_mask_fraction: float = 0.25,
        dilate: int = 15,
        fill: str = "patch",
        batch_size: int = 16,
        num_workers: int = 8,
        path: str | None = None,
        seed: int = 0,
        **folder_kwargs,
    ) -> None:
        self.dataset_root = path or os.path.join(DATASETS_PATH, NAME)
        fetch_blob(self.dataset_root, kaggle=KAGGLE)

        root = find_root(self.dataset_root)
        tag = "all" if not categories else "-".join(sorted(categories))
        benchmark_root = os.path.join(root, "anomaly_detection", f"{tag}_{max_images}")
        if not os.path.exists(os.path.join(benchmark_root, "test", "norm")):
            generate_sardet(root, benchmark_root, max_images, categories, min_box, max_mask_fraction, dilate, fill, seed)

        batch_size = 1 if DEBUG else batch_size
        super().__init__(
            name=NAME,
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
