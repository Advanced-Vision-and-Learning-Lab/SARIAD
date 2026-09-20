"""Generate "normal" (target-free) SAR images from images that contain targets.

Anomaly detection benchmarks need normal training data, but SAR target datasets only contain
images *with* targets. Normal images are synthesized in two steps that this module separates so
that each can be swapped:

1. **Segment** the target (and its shadow) -> binary mask. Backends live in :data:`SEGMENTERS`
   (``"classical"``: KMeans + morphology; ``"birefnet"``: a learned salient-object model).
   Datasets that ship ground-truth masks (e.g. SSDD) skip this step.
2. **Fill** the masked pixels with clutter taken from the rest of the image:
   :func:`fill_from_background` (i.i.d. random background pixels) or :func:`patch_inpaint`
   (random background patches, better for structured clutter such as sea).

:func:`generate_normal` runs step 2, :func:`segment_target` step 1.

Example:
    >>> mask = segment_target(image)                        # (H, W) uint8, 1 = target
    >>> normal = generate_normal(image, mask, method="background")
"""

import numpy as np

__all__ = [
    "SEGMENTERS",
    "SegmentationFailed",
    "remove_target",
    "blur_mask",
    "dilate_mask",
    "fill_from_background",
    "generate_normal",
    "patch_inpaint",
    "segment_target",
]


class SegmentationFailed(ValueError):
    """The segmentation is not trustworthy (empty, or covering most of the image)."""


def _rng(rng: np.random.Generator | int | None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


# ---------------------------------------------------------------------------- mask utilities
def blur_mask(mask: np.ndarray, sigma: float = 1.0, threshold: float = 0.15) -> np.ndarray:
    """Grow/smooth a binary mask: Gaussian blur, then threshold back to ``{0, 1}``.

    A low ``threshold`` (default ``0.15``) makes the mask expand by a few ``sigma``.
    """
    from scipy.ndimage import gaussian_filter

    return (gaussian_filter(mask.astype(float), sigma=sigma) > threshold).astype(np.uint8)


def dilate_mask(mask: np.ndarray, size: int) -> np.ndarray:
    """Dilate a binary mask with a ``size x size`` square, e.g. to also cover the target's smeared edges."""
    import cv2

    if size <= 1:
        return (mask > 0).astype(np.uint8)
    return (cv2.dilate((mask > 0).astype(np.uint8), np.ones((size, size), np.uint8)) > 0).astype(np.uint8)


# ---------------------------------------------------------------------------- segmentation
def segment_classical(
    image: np.ndarray,
    n_clusters: int = 2,
    sigma: float = 4,
    kernel_size: int = 50,
    shadow: bool = False,
    seed: int | None = None,
) -> np.ndarray:
    """Segment the brightest (or, with ``shadow``, the darkest) structure with KMeans on the smoothed image.

    Pixels are clustered by intensity after Gaussian smoothing; the darkest cluster is background.
    The target mask is closed morphologically (``kernel_size``, 0 to skip) and only the largest
    connected component is kept. With ``shadow=True`` the roles are inverted, which finds the radar
    shadow of a target.

    Args:
        image: ``(H, W)`` or ``(H, W, 1)`` float intensity image.
        n_clusters: KMeans clusters.
        sigma: Std. dev. of the Gaussian pre-smoothing.
        kernel_size: Diameter of the elliptical closing element (0 disables closing).
        shadow: Segment the shadow instead of the target.
        seed: KMeans seed (``None``: not reproducible).

    Returns:
        ``(H, W)`` uint8 mask, 1 on the segmented structure.
    """
    import cv2
    from scipy.ndimage import gaussian_filter
    from sklearn.cluster import KMeans

    gray = image[..., 0] if image.ndim == 3 else image
    smoothed = gaussian_filter(gray, sigma=sigma)
    values = smoothed.reshape(-1, 1)

    labels = KMeans(n_clusters=n_clusters, n_init=5, max_iter=100, random_state=seed).fit(values).labels_
    background = int(np.argmin([values[labels == i].mean() for i in range(n_clusters)]))
    mask = (labels.reshape(gray.shape) != background).astype(np.uint8)
    if shadow:
        mask = 1 - mask

    if kernel_size:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))

    count, components = cv2.connectedComponents(mask)
    if count <= 1:
        return np.zeros(gray.shape, np.uint8)
    sizes = np.bincount(components.ravel())[1:]  # skip label 0 (background)
    return (components == (np.argmax(sizes) + 1)).astype(np.uint8)


_BIREFNET = None


def segment_birefnet(image: np.ndarray, threshold: float = 0.5, size: int = 1024) -> np.ndarray:
    """Segment the salient object with BiRefNet (https://github.com/ZhengPeng7/BiRefNet).

    Experimental: BiRefNet is trained on natural RGB images, so how well it transfers to SAR
    intensity images is unverified. Requires ``pip install transformers`` and downloads the model from the
    Hugging Face hub with ``trust_remote_code=True`` (it executes code from that repository).

    Args:
        image: ``(H, W)`` or ``(H, W, 1)`` float image in ``[0, 1]``.
        threshold: Probability above which a pixel is target.
        size: Side length the image is resized to for the network.
    """
    global _BIREFNET
    import torch
    import torch.nn.functional as F  # noqa: N812

    if _BIREFNET is None:
        from transformers import AutoModelForImageSegmentation

        _BIREFNET = AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).eval()
    gray = image[..., 0] if image.ndim == 3 else image
    x = torch.from_numpy(np.clip(gray, 0, 1).astype(np.float32))[None, None].repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    with torch.no_grad():
        prob = _BIREFNET(x)[-1].sigmoid()
    prob = F.interpolate(prob, size=gray.shape, mode="bilinear", align_corners=False)[0, 0]
    return (prob.numpy() > threshold).astype(np.uint8)


SEGMENTERS = {"classical": segment_classical, "birefnet": segment_birefnet}


def segment_target(image: np.ndarray, backend: str = "classical", **kwargs) -> np.ndarray:
    """Binary target mask of ``image`` (``(H, W)`` uint8, 1 = target) with the chosen backend."""
    if backend not in SEGMENTERS:
        msg = f"Unknown segmentation backend {backend!r}. Available: {sorted(SEGMENTERS)}"
        raise ValueError(msg)
    return SEGMENTERS[backend](image, **kwargs)


# ---------------------------------------------------------------------------- filling
def fill_from_background(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Replace every masked pixel by a randomly chosen unmasked pixel of the same image.

    Works for ``(H, W)`` and ``(H, W, C)`` images. Returns the image unchanged if everything is masked.
    """
    background = mask == 0
    masked = ~background
    output = image.copy()
    pool = image[background]
    if len(pool) == 0 or not masked.any():
        return output
    output[masked] = pool[_rng(rng).integers(len(pool), size=int(masked.sum()))]
    return output


def patch_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    min_crop_size: int = 3,
    max_crop_size: int = 15,
    sample_step: int = 10,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Fill the masked region with random square patches of background (grayscale ``(H, W)`` or BGR image).

    Background patches (side ``min_crop_size..max_crop_size``, on a ``sample_step`` grid, fully outside the
    mask) are pooled, keeping only those on the "clutter side" of the image mean (darker patches for dark
    images, brighter for bright ones) so that no target-like bright returns are pasted into dark sea. Patches
    are pasted around the masked pixels in random order and overlaps averaged.

    Returns:
        Image of the same dtype and shape as ``image`` (grayscale).
    """
    import cv2

    rng = _rng(rng)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = image.astype(np.float32)
    mask = mask.astype(np.uint8)
    height, width = mask.shape
    mean = img[mask == 0].mean() if (mask == 0).any() else img.mean()
    keep = (lambda m: m >= mean) if mean / 255.0 >= 0.5 else (lambda m: m <= mean)

    pool = []
    for top in range(0, height - min_crop_size + 1, sample_step):
        for left in range(0, width - min_crop_size + 1, sample_step):
            fits_max = top + max_crop_size <= height and left + max_crop_size <= width
            if fits_max and not np.all(mask[top : top + max_crop_size, left : left + max_crop_size] == 0):
                continue  # the largest window touches the mask: no patch from here
            for k in range(min_crop_size, max_crop_size + 1):
                if top + k > height or left + k > width:
                    continue
                if not fits_max and np.any(mask[top : top + k, left : left + k] != 0):
                    continue
                patch = img[top : top + k, left : left + k]
                if keep(patch.mean()):
                    pool.append(patch)
    if not pool:
        return image

    working = mask.copy()
    canvas = np.zeros_like(img)
    weight = np.zeros(img.shape, np.int32)
    ys, xs = np.nonzero(mask)
    for i in rng.permutation(len(ys)):
        cy, cx = ys[i], xs[i]
        if working[cy, cx] == 0:
            continue
        patch = pool[int(rng.integers(len(pool)))]
        ph, pw = patch.shape
        y0, x0 = cy - ph // 2, cx - pw // 2
        y1, x1 = max(0, y0), max(0, x0)
        y2, x2 = min(height, y0 + ph), min(width, x0 + pw)
        piece = patch[y1 - y0 : y2 - y0, x1 - x0 : x2 - x0]
        region = working[y1:y2, x1:x2] == 1
        if region.any():
            canvas[y1:y2, x1:x2][region] += piece[region]
            weight[y1:y2, x1:x2][region] += 1
            working[y1:y2, x1:x2][region] = 0

    weight[weight == 0] = 1
    output = img.copy()
    output[mask == 1] = (canvas / weight)[mask == 1]
    return np.clip(output, 0, 255).astype(image.dtype)


def generate_normal(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "background",
    dilate: int = 0,
    rng: np.random.Generator | int | None = None,
    **kwargs,
) -> np.ndarray:
    """Remove the masked target(s) from ``image``.

    Args:
        image: Image containing targets.
        mask: Binary target mask (non-zero = target).
        method: ``"background"`` (:func:`fill_from_background`) or ``"patch"`` (:func:`patch_inpaint`).
        dilate: Dilate the mask by a ``dilate x dilate`` square first (``0``: no dilation).
        rng: Seed or generator, for reproducible results.
        **kwargs: Passed to the fill method.
    """
    mask = dilate_mask(mask, dilate) if dilate else (mask > 0).astype(np.uint8)
    if method == "background":
        return fill_from_background(image, mask, rng)
    if method == "patch":
        return patch_inpaint(image, mask, rng=rng, **kwargs)
    msg = f"Unknown method {method!r}: use 'background' or 'patch'"
    raise ValueError(msg)


def remove_target(
    image: np.ndarray,
    segmenter: str = "classical",
    shadow_sigma: float = 10,
    rng: np.random.Generator | int | None = None,
    max_mask_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a target chip into a normal image: remove the target *and* its radar shadow.

    This is the recipe used for MSTAR-like chips (one centered target on clutter): segment and
    blur the target, fill it with background pixels, then segment the (now more prominent) shadow
    on the result, blur it and fill that too.

    Args:
        image: ``(H, W)`` or ``(H, W, 1)`` float intensity image in ``[0, 1]``.
        segmenter: Backend for the target mask, see :data:`SEGMENTERS`.
        shadow_sigma: Smoothing of the shadow segmentation (larger for smoother clutter).
        rng: Seed or generator for the background fill.
        max_mask_fraction: Largest fraction of the image the target (or target + shadow) may cover.

    Returns:
        ``(normal_image, mask)``: the image without target/shadow (same shape as ``image``) and the
        ``(H, W)`` uint8 mask (1 on target and shadow) that was removed.

    Raises:
        SegmentationFailed: If the target mask is empty or larger than ``max_mask_fraction``. The
            classical segmenter assumes a dark clutter background; on a brighter or noisier chip it can
            select half the image, and the "normal" image would still contain the target.
    """
    rng = _rng(rng)
    target = blur_mask(segment_target(image, backend=segmenter), sigma=13)
    if not 0 < target.mean() <= max_mask_fraction:
        msg = f"Target mask covers {target.mean():.0%} of the image (allowed: 0-{max_mask_fraction:.0%}]"
        raise SegmentationFailed(msg)
    without_target = fill_from_background(image, target, rng)
    shadow = segment_target(without_target, backend="classical", n_clusters=5, sigma=shadow_sigma, shadow=True, kernel_size=0)
    shadow = blur_mask(shadow, sigma=10)
    combined = np.maximum(target, shadow)
    if combined.mean() > max_mask_fraction:
        msg = f"Target + shadow mask covers {combined.mean():.0%} of the image (allowed: up to {max_mask_fraction:.0%})"
        raise SegmentationFailed(msg)
    normal = fill_from_background(without_target, shadow, rng)
    return np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0), combined
