"""YOLOAnomaly: anomaly detection with features from Ultralytics models.

Hierarchical features of a (pretrained) Ultralytics backbone are collected on normal images, a
multivariate Gaussian is fitted to them and test images are scored with the Mahalanobis
distance (PaDiM with a YOLO-family backbone). No gradient training is involved.

Example:
    >>> from SARIAD.datasets import MSTAR
    >>> from SARIAD.models import YOLOAnomaly
    >>> from anomalib.engine import Engine

    >>> engine = Engine()
    >>> model = YOLOAnomaly(weights="yolo11n.pt")
    >>> engine.fit(model=model, datamodule=MSTAR())
"""

from torch import nn
from torchvision.transforms.v2 import Compose, Resize

from anomalib.metrics import Evaluator
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer
from SARIAD.models.components import GaussianAD

from .torch_model import DEFAULT_LAYERS, YOLOAnomalyModel

__all__ = ["YOLOAnomaly"]


class YOLOAnomaly(GaussianAD):
    """Gaussian anomaly detector on Ultralytics (YOLO, RT-DETR, FastSAM, ...) features.

    Args:
        weights (str, optional): Ultralytics weights. Defaults to ``"yolo11n.pt"``.
        layers (tuple[int, ...], optional): Layers of ``YOLO(weights).model.model`` to use.
            Defaults to the three backbone stages of YOLOv8/YOLO11.
        n_features (int | None, optional): Embedding dimensions kept. Defaults to 100.
        pre_processor, post_processor, evaluator, visualizer: As for any anomalib model.
    """

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        layers: tuple[int, ...] = DEFAULT_LAYERS,
        n_features: int | None = None,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            YOLOAnomalyModel(weights=weights, layers=tuple(layers), n_features=n_features),
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

    @staticmethod
    def configure_pre_processor(image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Resize (to a multiple of 32, the largest stride). Ultralytics models take [0, 1] RGB, so no normalization."""
        return PreProcessor(Compose([Resize(image_size or (256, 256), antialias=True)]))

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        return PostProcessor()
