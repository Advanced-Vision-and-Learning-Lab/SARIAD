"""Filter augmentation for SAR images (the input stage of MSFA).

MSFA ("Multi-Stage with Filter Augmentation", Li et al., SARDet-100K, NeurIPS 2024,
https://arxiv.org/abs/2403.06534) does not feed a SAR image to a backbone as a replicated grayscale
image. It stacks the raw intensity with hand-crafted filter responses that capture the structure of
SAR imagery (edges robust to speckle, gradient orientation histograms, ...) and widens the first
layer of the backbone to accept them.

The reference implementation lives in an MMDetection project and uses CUDA-only operations;
these are dependency-free, device-agnostic re-implementations of a subset of its filters:

=============  ========  ====================================================================
name           channels  description
=============  ========  ====================================================================
``raw``        1         intensity (mean of the input channels)
``grad_edge``  1         log-ratio-of-averages edge strength (constant false alarm rate style)
``hog``        9         histogram of oriented gradients, per pixel, 8x8 cells, bilinear upsampled
``canny``      6         blurred image, gradient magnitude and orientation, thin edges (non-maximum
                         suppression), thresholded edges, thresholded gradient magnitude
=============  ========  ====================================================================

The wavelet-scattering and Haar filters of the original need extra dependencies
(``kymatio``, ``torchhaarfeatures``) and are not included.

All filters expect intensities in ``[0, 1]`` (do not mean/std-normalize before them: the
log-ratio needs positive values).
"""

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

FILTER_CHANNELS = {"raw": 1, "grad_edge": 1, "hog": 9, "canny": 6}


class GradEdge(nn.Module):
    """Log-ratio-of-averages edge strength: ``|(log(mean_a / mean_b))|`` over opposite half windows, x and y."""

    def __init__(self, radius: int = 5) -> None:
        super().__init__()
        r, size = radius, 2 * radius + 1
        left = torch.cat([torch.ones(size, r + 1), torch.zeros(size, r)], dim=1)
        top = torch.cat([torch.ones(r + 1, size), torch.zeros(r, size)], dim=0)
        self.radius = radius
        self.register_buffer("kernels", torch.stack([left, left.flip(1), top, top.flip(0)]).unsqueeze(1))  # (4, 1, s, s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, H, W) -> (B, 1, H, W)
        r = self.radius
        x = F.pad(x, (r, r, r, r), mode="reflect") + 1e-2  # offset keeps the ratio finite on dark pixels
        left, right, top, bottom = F.conv2d(x, self.kernels).unbind(dim=1)
        gx, gy = torch.log(left / right), torch.log(top / bottom)
        return torch.sqrt(gx**2 + gy**2).unsqueeze(1)


class HOG(nn.Module):
    """Per-pixel histogram of oriented gradients (``nbins`` orientations, ``pool x pool`` cells)."""

    def __init__(self, nbins: int = 9, pool: int = 8) -> None:
        super().__init__()
        self.nbins, self.pool = nbins, pool
        sobel = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        self.register_buffer("kernels", torch.stack([sobel, sobel.T]).unsqueeze(1))  # (2, 1, 3, 3): x, y

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, H, W) -> (B, nbins, H, W)
        height, width = x.shape[-2:]
        gx, gy = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), self.kernels).unbind(dim=1)
        magnitude = torch.sqrt(gx**2 + gy**2)
        bins = (torch.atan2(gx, gy) / math.pi * self.nbins).floor().long() % self.nbins  # (B, H, W)
        hist = torch.zeros(x.shape[0], self.nbins, height, width, device=x.device, dtype=x.dtype)
        hist.scatter_add_(1, bins.unsqueeze(1), magnitude.unsqueeze(1))
        hist = F.avg_pool2d(hist, self.pool) * self.pool**2  # sum over cells
        return F.interpolate(hist, size=(height, width), mode="bilinear", align_corners=False)


class Canny(nn.Module):
    """Device-agnostic Canny features: returns 6 channels (see module docstring)."""

    def __init__(self, threshold: float = 0.05) -> None:
        super().__init__()
        self.threshold = threshold
        g = torch.exp(-0.5 * (torch.arange(5.0) - 2.0) ** 2)
        g = g / g.sum()
        sobel = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        self.register_buffer("gauss", (g[:, None] * g[None, :]).view(1, 1, 5, 5))
        self.register_buffer("sobel", torch.stack([sobel, sobel.T]).unsqueeze(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 1, H, W) -> (B, 6, H, W)
        blurred = F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), self.gauss)
        gx, gy = F.conv2d(F.pad(blurred, (1, 1, 1, 1), mode="reflect"), self.sobel).unbind(dim=1)
        magnitude = torch.sqrt(gx**2 + gy**2).unsqueeze(1)
        angle = (torch.rad2deg(torch.atan2(gy, gx)) % 180.0).unsqueeze(1)  # gradient direction in [0, 180)
        direction = torch.round(angle / 45.0).long() % 4  # 0: horizontal, 1: 45deg, 2: vertical, 3: 135deg

        padded = F.pad(magnitude, (1, 1, 1, 1), mode="replicate")
        height, width = magnitude.shape[-2:]

        def shifted(dy: int, dx: int) -> torch.Tensor:
            return padded[..., 1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]

        # neighbours along the gradient direction (y down): 0 -> (0, +-1), 1 -> (-1,+1)/(+1,-1), 2 -> (+-1, 0), 3 -> (-1,-1)/(+1,+1)
        offsets = [((0, 1), (0, -1)), ((-1, 1), (1, -1)), ((1, 0), (-1, 0)), ((-1, -1), (1, 1))]
        keep = torch.zeros_like(magnitude, dtype=torch.bool)
        for d, (a, b) in enumerate(offsets):
            keep |= (direction == d) & (magnitude >= shifted(*a)) & (magnitude >= shifted(*b))
        thin = magnitude * keep
        thresholded = (thin >= self.threshold).to(x.dtype)
        early = magnitude * (magnitude >= self.threshold)
        return torch.cat([blurred, magnitude, angle, thin, thresholded, early], dim=1)


class FilterAugmentation(nn.Module):
    """Stack of filter responses of a SAR image, standardized per image and channel.

    Args:
        filters: Names from :data:`FILTER_CHANNELS`, in output order. ``"raw"`` is normally first.
        canny_threshold: Edge threshold of the ``canny`` filter (intensities in ``[0, 1]``).
    """

    def __init__(self, filters: tuple[str, ...] = ("raw", "grad_edge", "hog"), canny_threshold: float = 0.05) -> None:
        super().__init__()
        unknown = [f for f in filters if f not in FILTER_CHANNELS]
        if unknown or not filters:
            msg = f"filters must be a non-empty subset of {sorted(FILTER_CHANNELS)}, got {list(filters)}"
            raise ValueError(msg)
        self.filters = tuple(filters)
        self.out_channels = sum(FILTER_CHANNELS[f] for f in filters)
        self.grad_edge = GradEdge() if "grad_edge" in filters else None
        self.hog = HOG() if "hog" in filters else None
        self.canny = Canny(canny_threshold) if "canny" in filters else None

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` in ``[0, 1]`` -> ``(B, out_channels, H, W)``."""
        gray = images.mean(dim=1, keepdim=True)
        responses = []
        for name in self.filters:
            responses.append(
                gray if name == "raw" else getattr(self, name)(gray),
            )
        stack = torch.cat(responses, dim=1)
        mean = stack.mean(dim=(-2, -1), keepdim=True)
        std = stack.std(dim=(-2, -1), keepdim=True)
        return (stack - mean) / (std + 1e-6)
