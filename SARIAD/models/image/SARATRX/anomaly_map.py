"""Anomaly map generator for the SARATRX model.

SARATR-X's masked autoencoder does not reconstruct pixels: for every 16x16 patch its decoder
predicts three multi-scale SAR gradient features (the ``sarfeature{1,2,3}`` maps of the
submodule, each patchified and, with ``norm_pix_loss``, normalized per patch). The anomaly
signal is therefore the error between the *predicted* and the *target* features, computed on
the patches that were masked out.

Since the masking is random, an image is scored with several masks and the error of every
patch is averaged over the passes in which that patch was masked, so that all patches
contribute (a single pass at ``mask_ratio=0.75`` would only cover about 75% of them).

Example:
    >>> generator = AnomalyMapGenerator(sigma=4)
    >>> pred = torch.randn(2, 196, 768)     # decoder output, 3 scales x 16 x 16 per patch
    >>> target = torch.randn(2, 196, 768)
    >>> error = generator.patch_error(pred, target)                     # (2, 196, 256)
    >>> mask = torch.randint(0, 2, (2, 196)).float()
    >>> anomaly_map = generator(error * mask[..., None], mask, image_size=224, patch_size=16)
    >>> anomaly_map.shape
    torch.Size([2, 1, 224, 224])
"""

import torch
from torch import nn

from anomalib.models.components import GaussianBlur2d


class AnomalyMapGenerator(nn.Module):
    """Turn patch-level feature prediction errors into a smoothed pixel-level anomaly map.

    Args:
        sigma (int, optional): Standard deviation of the Gaussian smoothing kernel.
            Defaults to ``4``.
        error_mode (str, optional): ``"l2"`` (squared error) or ``"l1"`` (absolute error).
            Defaults to ``"l2"``.
    """

    def __init__(self, sigma: int = 4, error_mode: str = "l2") -> None:
        super().__init__()
        if error_mode not in ("l1", "l2"):
            msg = f"error_mode must be 'l1' or 'l2', got {error_mode!r}"
            raise ValueError(msg)
        self.error_mode = error_mode
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        self.blur = GaussianBlur2d(kernel_size=(kernel_size, kernel_size), sigma=(sigma, sigma), channels=1)

    def patch_error(self, pred: torch.Tensor, target: torch.Tensor, num_scales: int = 3) -> torch.Tensor:
        """Per-pixel error of every patch, averaged over the feature scales.

        Args:
            pred: Decoder output, ``(B, L, S * p * p)``: ``S`` scales, each a flattened ``p x p`` patch.
            target: Target features in the same layout and normalization as ``pred``.
            num_scales: Number of feature scales ``S`` (3 for SARATR-X).

        Returns:
            Tensor of shape ``(B, L, p * p)``.
        """
        diff = pred - target
        error = diff.abs() if self.error_mode == "l1" else diff**2
        batch, num_patches, dim = error.shape
        return error.reshape(batch, num_patches, num_scales, dim // num_scales).mean(dim=2)

    @staticmethod
    def unpatch(patches: torch.Tensor, patch_size: int, image_size: int) -> torch.Tensor:
        """Reassemble ``(B, L, p * p)`` patches (row-major, as produced by ``patchify``) into ``(B, 1, H, W)``."""
        batch = patches.shape[0]
        side = image_size // patch_size
        patches = patches.reshape(batch, side, side, patch_size, patch_size)
        return patches.permute(0, 1, 3, 2, 4).reshape(batch, 1, image_size, image_size)

    def forward(self, error_sum: torch.Tensor, mask_count: torch.Tensor, image_size: int, patch_size: int = 16) -> torch.Tensor:
        """Average the accumulated error over the passes in which each patch was masked.

        Args:
            error_sum: ``(B, L, p * p)``: error summed over the passes, only counting masked patches.
            mask_count: ``(B, L)``: in how many passes each patch was masked.
            image_size: Side length of the (square) input image.
            patch_size: Patch side length. Defaults to ``16``.

        Returns:
            Smoothed anomaly map of shape ``(B, 1, image_size, image_size)``.
        """
        error = error_sum / mask_count.clamp_min(1).unsqueeze(-1)
        return self.blur(self.unpatch(error, patch_size, image_size))
