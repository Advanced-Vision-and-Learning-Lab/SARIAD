import numpy as np
import pytest

from SARIAD.utils import normal_gen as ng


@pytest.fixture
def chip():
    rng = np.random.default_rng(0)
    image = rng.gamma(2.0, 0.05, (100, 100)).astype(np.float32)
    yy, xx = np.mgrid[:100, :100]
    image[((yy - 45) ** 2 / 300 + (xx - 40) ** 2 / 120) < 1] += 0.9          # bright target
    image[((yy - 45) ** 2 / 300 + (xx - 62) ** 2 / 60) < 1] *= 0.1          # its shadow
    return image[..., None]


def test_segmentation_is_reproducible_and_finds_the_target(chip):
    mask = ng.segment_target(chip, seed=0)
    assert mask.shape == (100, 100) and mask.dtype == np.uint8
    assert np.array_equal(mask, ng.segment_target(chip, seed=0))
    assert mask[45, 40] == 1 and mask[5, 5] == 0                                 # target yes, corner no


def test_unknown_backend():
    with pytest.raises(ValueError, match="Unknown segmentation backend"):
        ng.segment_target(np.zeros((8, 8), np.float32), backend="nope")


def test_fill_from_background_only_touches_masked_pixels_and_uses_background_values(chip):
    mask = np.zeros((100, 100), np.uint8)
    mask[30:60, 30:60] = 1
    filled = ng.fill_from_background(chip, mask, rng=0)
    assert np.array_equal(filled[mask == 0], chip[mask == 0])
    assert np.isin(filled[mask == 1], chip[mask == 0]).all()
    assert np.array_equal(filled, ng.fill_from_background(chip, mask, rng=0))     # reproducible
    assert np.array_equal(ng.fill_from_background(chip, np.ones((100, 100), np.uint8)), chip)  # nothing to draw from


def test_patch_inpaint_keeps_unmasked_pixels_and_dtype():
    rng = np.random.default_rng(1)
    gray = (rng.gamma(2.0, 0.05, (120, 120)).clip(0, 1) * 255).astype(np.uint8)
    mask = np.zeros((120, 120), np.uint8)
    mask[40:70, 45:80] = 1
    out = ng.patch_inpaint(gray, mask, rng=0)
    assert out.dtype == np.uint8 and out.shape == gray.shape
    assert np.array_equal(out[mask == 0], gray[mask == 0])
    assert abs(float(out[mask == 1].mean()) - float(gray[mask == 0].mean())) < 10
    assert ng.patch_inpaint(np.stack([gray] * 3, -1), mask, rng=0).shape == (120, 120)   # BGR input -> gray


def test_generate_normal_dilate_and_method_validation():
    gray = np.full((60, 60), 100, np.uint8)
    mask = np.zeros((60, 60), np.uint8)
    mask[25:35, 25:35] = 1
    gray[mask == 1] = 250
    out = ng.generate_normal(gray, mask, method="background", dilate=5, rng=0)
    assert (out == 250).sum() == 0
    with pytest.raises(ValueError, match="Unknown method"):
        ng.generate_normal(gray, mask, method="x")


def test_remove_target_removes_target_and_reports_mask(chip):
    normal, mask = ng.remove_target(chip, rng=0)
    assert normal.shape == chip.shape and mask.shape == (100, 100)
    assert 0 < mask.mean() < 0.5
    assert normal[..., 0][mask == 1].max() < chip[..., 0][mask == 1].max()          # the bright return is gone


def test_remove_target_rejects_untrustworthy_segmentation():
    """On a featureless chip KMeans splits the noise in half: that must not be written as a 'normal' image."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0.6, 0.05, (128, 128, 1)).astype(np.float32)
    with pytest.raises(ng.SegmentationFailed):
        ng.remove_target(noise, rng=0)
