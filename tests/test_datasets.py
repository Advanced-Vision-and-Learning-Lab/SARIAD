import json

import cv2
import numpy as np
import pytest

from SARIAD.datasets.datamodules.image.sample_public.sample_public import generate_sample
from SARIAD.datasets.datamodules.image.sardet.sardet import find_root, generate_sardet


def _speckled_chip(rng, size=128, target=True):
    base = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 5)
    chip = 0.08 + 0.05 * (base - base.min()) / (np.ptp(base) + 1e-6)
    chip = chip * rng.gamma(4.0, 1 / 4.0, (size, size))
    if target:
        yy, xx = np.mgrid[:size, :size]
        chip[((yy - 60) ** 2 / 250 + (xx - 50) ** 2 / 100) < 1] += 0.7
        chip[((yy - 60) ** 2 / 250 + (xx - 72) ** 2 / 50) < 1] *= 0.05
    return (chip.clip(0, 1) * 255).astype(np.uint8)


def test_generate_sample_splits_by_azimuth_and_writes_triplets(tmp_path):
    rng = np.random.default_rng(0)
    src = tmp_path / "src"
    for azimuth in (10, 50, 100, 200, 300, 340):
        directory = src / "png_images" / "qpm" / "real" / "2s1"
        directory.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(directory / f"2s1_real_A_elevDeg_015_azCenter_{azimuth:03d}_22_serial_b01.png"), _speckled_chip(rng))
    counts = generate_sample(src, tmp_path / "out", "real", "qpm", test_azimuth_ratio=0.2, seed=0)
    assert counts["train"] == 4 and counts["test"] == 2                              # azimuth >= 288 -> test
    for split in ("train", "test"):
        names = {kind: sorted(p.name for p in (tmp_path / "out" / split / kind).iterdir()) for kind in ("anom", "norm", "masks")}
        assert names["anom"] == names["norm"] == names["masks"]
    anom = cv2.imread(str(next((tmp_path / "out/test/anom").iterdir())), 0)
    norm = cv2.imread(str(next((tmp_path / "out/test/norm").iterdir())), 0)
    assert anom.max() > norm.max() or anom.mean() > norm.mean()                     # the target is gone from the normal image


def test_generate_sample_skips_unsegmentable_chips_instead_of_writing_bad_normals(tmp_path):
    rng = np.random.default_rng(0)
    directory = tmp_path / "src" / "png_images" / "qpm" / "real" / "x"
    directory.mkdir(parents=True)
    for az in (20, 330):
        noise = (rng.normal(0.6, 0.05, (128, 128)).clip(0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(directory / f"x_real_azCenter_{az:03d}.png"), noise)
    with pytest.raises(RuntimeError, match="split is empty"):
        generate_sample(tmp_path / "src", tmp_path / "out", "real", "qpm", 0.2)


def _fake_sardet(root, rng):
    (root / "JPEGImages").mkdir(parents=True)
    (root / "Annotations").mkdir()

    def make(split, n, start):
        images, anns = [], []
        for i in range(n):
            image_id = start + i
            image = rng.gamma(2.0, 30, (160, 160)).clip(0, 255).astype(np.uint8)
            x, y = rng.integers(20, 100, 2)
            image[y : y + 24, x : x + 30] = 250
            cv2.imwrite(str(root / "JPEGImages" / f"{image_id}.jpg"), image)
            images.append({"id": image_id, "file_name": f"{image_id}.jpg"})
            anns.append({"id": image_id, "image_id": image_id, "category_id": 1 + i % 2, "bbox": [int(x), int(y), 30, 24]})
        cats = [{"id": 1, "name": "ship"}, {"id": 2, "name": "aircraft"}]
        (root / "Annotations" / f"{split}.json").write_text(json.dumps({"images": images, "annotations": anns, "categories": cats}))

    make("train", 12, 0)
    make("test", 6, 1000)


def test_generate_sardet_from_coco(tmp_path):
    root = tmp_path / "download" / "SARDet_100K"                                    # nested like a Kaggle archive
    _fake_sardet(root, np.random.default_rng(0))
    assert find_root(tmp_path / "download") == root
    written = generate_sardet(root, tmp_path / "out", max_images=4, categories=["ship"], dilate=9, fill="background", seed=0)
    assert written == {"train": 4, "test": 3}                                       # capped / only the ship annotations
    mask = cv2.imread(str(next((tmp_path / "out/test/masks").iterdir())), 0)
    anom = cv2.imread(str(next((tmp_path / "out/test/anom").iterdir())), 0)
    norm = cv2.imread(str(tmp_path / "out/test/norm" / next((tmp_path / "out/test/anom").iterdir()).name), 0)
    assert mask.max() == 255 and (anom >= 250).sum() > (norm >= 250).sum() + 300     # the object was filled with clutter
    with pytest.raises(ValueError, match="None of"):
        generate_sardet(root, tmp_path / "out2", categories=["submarine"])


def test_find_root_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_root(tmp_path)
