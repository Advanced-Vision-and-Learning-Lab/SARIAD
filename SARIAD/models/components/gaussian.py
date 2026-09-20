"""PaDiM-style Gaussian anomaly detection on top of an arbitrary feature extractor.

Anomalib's :class:`~anomalib.models.image.padim.torch_model.PadimModel` fits a multivariate
Gaussian to hierarchical CNN features of normal images and scores test images by the
Mahalanobis distance to it. It is hard-wired to a timm backbone; the classes here keep all of
its behavior (embedding subsampling, Gaussian fit, anomaly map) but take the feature extractor
as an argument, so any model that can produce feature maps can be turned into an anomaly
detector (see the YOLO and MSFA models).

A feature extractor is an ``nn.Module`` that maps a ``(B, C, H, W)`` batch to a
``dict[str, Tensor]`` of ``(B, C_i, H_i, W_i)`` feature maps and has an ``out_dims`` attribute
listing the ``C_i`` in the same order.
"""

import torch
from torch import nn

from anomalib.data.utils.tiler import Tiler  # noqa: F401  (type of PadimModel.tiler)
from anomalib.models.components import MultiVariateGaussian
from anomalib.models.image.padim import Padim
from anomalib.models.image.padim.anomaly_map import AnomalyMapGenerator
from anomalib.models.image.padim.torch_model import PadimModel

__all__ = ["FeatureGaussianModel", "GaussianAD"]


class FeatureGaussianModel(PadimModel):
    """``PadimModel`` with a caller-supplied feature extractor.

    Args:
        feature_extractor: Module returning a dict of feature maps and exposing ``out_dims``.
        layers: Keys of the feature dict to use, in the order they are concatenated. The maps are
            resized to the resolution of the first one.
        n_features: Number of embedding dimensions kept (randomly selected once, like PaDiM) to
            keep the per-position covariance tractable. ``None`` keeps ``min(100, all)``.
    """

    def __init__(self, feature_extractor: nn.Module, layers: list[str], n_features: int | None = None) -> None:
        nn.Module.__init__(self)  # deliberately skip PadimModel.__init__, which builds a timm extractor
        self.tiler = None
        self.backbone = type(feature_extractor).__name__
        self.layers = list(layers)
        self.feature_extractor = feature_extractor.eval()
        self.n_features_original = sum(self.feature_extractor.out_dims)
        self.n_features = min(n_features or 100, self.n_features_original)
        if self.n_features <= 0:
            msg = f"n_features must be positive, got {n_features}"
            raise ValueError(msg)

        # randomly selected once and saved with the model, so results are reproducible
        self.register_buffer("idx", torch.randperm(self.n_features_original)[: self.n_features])
        self.loss = None
        self.anomaly_map_generator = AnomalyMapGenerator()
        self.gaussian = MultiVariateGaussian()
        self.memory_bank: list[torch.Tensor] = []

    def train(self, mode: bool = True) -> "FeatureGaussianModel":
        """The feature extractor is frozen: keep it in eval mode (e.g. batch-norm statistics)."""
        super().train(mode)
        self.feature_extractor.eval()
        return self


class GaussianAD(Padim):
    """Lightning wrapper for a :class:`FeatureGaussianModel`.

    Reuses the fit/validation logic of anomalib's ``Padim``: one pass over the normal training
    images collects embeddings, the Gaussian is fitted before validation.
    """

    def __init__(
        self,
        model: FeatureGaussianModel,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: nn.Module | bool = True,
        visualizer: nn.Module | bool = True,
    ) -> None:
        # Skip Padim.__init__ (it would build its own PadimModel) but run the base initializers.
        super(Padim, self).__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        self.model = model
