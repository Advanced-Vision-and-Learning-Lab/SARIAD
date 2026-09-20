"""PyTorch model for MSFA: filter-augmented features + a Gaussian anomaly model.

The image is turned into a stack of SAR filter responses (:mod:`.filters`), passed through a
timm backbone whose first layer is widened to that many channels (timm adapts the pretrained
weights), and the hierarchical features of normal images are modeled per patch with a
multivariate Gaussian (PaDiM); see :class:`~SARIAD.models.components.FeatureGaussianModel`.

Only the *data input* stage of the MSFA framework is reproduced here. Its other stages
(pretraining the backbone on SARDet-100K, then migrating it to the target detector) train
detection networks with MMDetection and are out of scope for an anomaly detector: pass SAR-pretrained
backbone weights as ``backbone`` (a timm model name or a module) to benefit from them.
"""

import timm
import torch
from torch import nn

from SARIAD.models.components import FeatureGaussianModel

from .filters import FilterAugmentation


class MSFAFeatureExtractor(nn.Module):
    """Filter augmentation followed by a multi-scale timm backbone.

    Args:
        backbone: timm model name (e.g. ``"resnet18"``).
        out_indices: Backbone stages to use (``features_only`` indices). Defaults to ``(1, 2, 3)``.
        filters: Filters to stack, see :class:`~.filters.FilterAugmentation`.
        pre_trained: Load pretrained weights. The first convolution is adapted to the number of filter channels.
        canny_threshold: Edge threshold of the ``canny`` filter.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        out_indices: tuple[int, ...] = (1, 2, 3),
        filters: tuple[str, ...] = ("raw", "grad_edge", "hog"),
        pre_trained: bool = True,
        canny_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        self.filters = FilterAugmentation(filters, canny_threshold)
        self.backbone = timm.create_model(
            backbone,
            pretrained=pre_trained,
            features_only=True,
            out_indices=tuple(out_indices),
            in_chans=self.filters.out_channels,
        ).eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.layer_names = [f"stage{i}" for i in out_indices]
        self.out_dims = list(self.backbone.feature_info.channels())

    def train(self, mode: bool = True) -> "MSFAFeatureExtractor":
        return super().train(False)  # frozen

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(self.filters(images))
        return dict(zip(self.layer_names, features, strict=True))


class MSFAModel(FeatureGaussianModel):
    """Gaussian anomaly model on filter-augmented backbone features.

    Args:
        backbone, out_indices, filters, pre_trained, canny_threshold: See :class:`MSFAFeatureExtractor`.
        n_features: Embedding dimensions kept (random subset). Defaults to 100.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        out_indices: tuple[int, ...] = (1, 2, 3),
        filters: tuple[str, ...] = ("raw", "grad_edge", "hog"),
        pre_trained: bool = True,
        canny_threshold: float = 0.05,
        n_features: int | None = None,
    ) -> None:
        extractor = MSFAFeatureExtractor(backbone, out_indices, filters, pre_trained, canny_threshold)
        super().__init__(extractor, extractor.layer_names, n_features)
