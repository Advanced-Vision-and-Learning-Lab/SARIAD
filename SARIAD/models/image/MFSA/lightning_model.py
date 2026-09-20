"""MSFA: SAR anomaly detection with filter-augmented inputs.

Inspired by the Multi-Stage with Filter Augmentation framework of SARDet-100K (Li et al., NeurIPS
2024): SAR filter responses (log-ratio edges, HOG, Canny, ...) are stacked with the raw intensity
and fed to a backbone, whose features of normal images are modeled with a per-patch Gaussian
(PaDiM). Scoring is the Mahalanobis distance; no gradient training is involved.

Example:
    >>> from SARIAD.datasets import MSTAR
    >>> from SARIAD.models import MSFA
    >>> from anomalib.engine import Engine

    >>> model = MSFA(filters=("raw", "grad_edge", "hog"))
    >>> Engine().fit(model=model, datamodule=MSTAR())
"""

from torch import nn
from torchvision.transforms.v2 import Compose, Resize

from anomalib.metrics import Evaluator
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer
from SARIAD.models.components import GaussianAD

from .torch_model import MSFAModel

__all__ = ["MSFA"]


class MSFA(GaussianAD):
    """Gaussian anomaly detector on filter-augmented SAR features.

    Args:
        backbone (str, optional): timm backbone. Defaults to ``"resnet18"``.
        out_indices (tuple[int, ...], optional): Backbone stages used. Defaults to ``(1, 2, 3)``.
        filters (tuple[str, ...], optional): Filters stacked as input: any of ``raw``,
            ``grad_edge``, ``hog``, ``canny``. Defaults to ``("raw", "grad_edge", "hog")``.
        pre_trained (bool, optional): Pretrained backbone weights. Defaults to ``True``.
        canny_threshold (float, optional): Edge threshold of the ``canny`` filter.
        n_features (int | None, optional): Embedding dimensions kept. Defaults to 100.
        pre_processor, post_processor, evaluator, visualizer: As for any anomalib model.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        out_indices: tuple[int, ...] = (1, 2, 3),
        filters: tuple[str, ...] = ("raw", "grad_edge", "hog"),
        pre_trained: bool = True,
        canny_threshold: float = 0.05,
        n_features: int | None = None,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            MSFAModel(
                backbone=backbone,
                out_indices=tuple(out_indices),
                filters=tuple(filters),
                pre_trained=pre_trained,
                canny_threshold=canny_threshold,
                n_features=n_features,
            ),
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

    @staticmethod
    def configure_pre_processor(image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Resize only: the filters need raw intensities in [0, 1], not mean/std-normalized images."""
        return PreProcessor(Compose([Resize(image_size or (256, 256), antialias=True)]))

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        return PostProcessor()
