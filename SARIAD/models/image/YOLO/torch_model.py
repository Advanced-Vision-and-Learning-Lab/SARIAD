"""Ultralytics backbones as anomaly-detection feature extractors.

YOLO is a supervised detector, so it can't flag anomalies directly. What it does offer is
features learned on large image collections: this module hooks intermediate layers of any
Ultralytics model (YOLOv5-v12, YOLO26, RT-DETR, FastSAM, ...) and hands their spatial feature
maps to a PaDiM-style Gaussian model (:class:`~SARIAD.models.components.FeatureGaussianModel`).

``Model.embed()`` is not used because it global-average-pools the features to a vector, which
would throw away the spatial layout needed for an anomaly map.

Example:
    >>> from SARIAD.models.image.YOLO.torch_model import YOLOAnomalyModel
    >>> model = YOLOAnomalyModel("yolo11n.pt")
    >>> model.train(); _ = model(torch.rand(8, 3, 256, 256)); model.fit()      # normal images
    >>> model.eval(); out = model(torch.rand(2, 3, 256, 256))                  # InferenceBatch
"""

import logging

import torch
from torch import nn

from SARIAD.models.components import FeatureGaussianModel

logger = logging.getLogger(__name__)

# Backbone stages (stride 8 / 16 / 32) of the YOLOv8 and YOLO11 families. Other architectures
# have different layer indices: inspect ``YOLO(weights).model.model`` and pass ``layers``.
DEFAULT_LAYERS = (4, 6, 9)


def _load_ultralytics_model(weights: str) -> nn.Module:
    """Load ``weights`` with the Ultralytics class that matches its name and return the ``nn.Module``."""
    from ultralytics import RTDETR, YOLO, FastSAM

    name = str(weights).lower()
    cls = RTDETR if "rtdetr" in name or "rt-detr" in name else FastSAM if "fastsam" in name else YOLO
    return cls(weights).model


class UltralyticsFeatureExtractor(nn.Module):
    """Spatial feature maps from intermediate layers of an Ultralytics model.

    Args:
        weights: Any Ultralytics weights/config name or path (e.g. ``"yolo11n.pt"``). Named weights are
            downloaded on first use.
        layers: Indices into ``model.model`` whose outputs are used (see :data:`DEFAULT_LAYERS`).
        input_size: Image size used to work out the channel dimension of every layer.
    """

    def __init__(self, weights: str = "yolo11n.pt", layers: tuple[int, ...] = DEFAULT_LAYERS, input_size: int = 256) -> None:
        super().__init__()
        self.net = _load_ultralytics_model(weights).float().eval()
        for param in self.net.parameters():
            param.requires_grad = False

        blocks = self.net.model
        for idx in layers:
            if not -len(blocks) <= idx < len(blocks):
                msg = f"Layer {idx} does not exist: {weights} has {len(blocks)} layers (0..{len(blocks) - 1})"
                raise ValueError(msg)
        self.layer_ids = [i % len(blocks) for i in layers]
        self.layer_names = [f"layer{i}" for i in self.layer_ids]

        # single-channel models (e.g. yolo11n-grayscale.pt) take the mean of the channels
        yaml = getattr(self.net, "yaml", None) or {}
        self.in_channels = int(yaml.get("channels", yaml.get("ch", 3)))

        self._features: dict[str, torch.Tensor] = {}
        for name, idx in zip(self.layer_names, self.layer_ids, strict=True):
            blocks[idx].register_forward_hook(self._make_hook(name))

        with torch.no_grad():
            dummy = self(torch.zeros(1, 3, input_size, input_size))
        bad = [n for n in self.layer_names if dummy[n].ndim != 4]
        if bad:
            msg = f"Layers {bad} of {weights} do not output (B, C, H, W) feature maps"
            raise ValueError(msg)
        self.out_dims = [dummy[n].shape[1] for n in self.layer_names]

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            self._features[name] = output

        return hook

    def train(self, mode: bool = True) -> "UltralyticsFeatureExtractor":
        return super().train(False)  # frozen: always in eval mode

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.in_channels == 1 and images.shape[1] != 1:
            images = images.mean(dim=1, keepdim=True)
        self._features = {}
        self.net(images)  # the detection head's output is discarded; the hooks collect the features
        features, self._features = self._features, {}
        return features


class YOLOAnomalyModel(FeatureGaussianModel):
    """Gaussian anomaly model on Ultralytics backbone features.

    Args:
        weights: Ultralytics weights (``"yolo11n.pt"``, ``"yolov8s.pt"``, ``"rtdetr-l.pt"``, ...).
        layers: Layer indices to use; the defaults suit YOLOv8/YOLO11.
        n_features: Embedding dimensions kept by the Gaussian model (random subset). Defaults to 100.
        input_size: Image size assumed when probing the layers.
    """

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        layers: tuple[int, ...] = DEFAULT_LAYERS,
        n_features: int | None = None,
        input_size: int = 256,
    ) -> None:
        extractor = UltralyticsFeatureExtractor(weights, layers, input_size)
        super().__init__(extractor, extractor.layer_names, n_features)
