"""PaDiM-ACE: patch distribution modeling with an adaptive cosine estimator for SAR anomaly detection.

Normal training images give the background statistics (as in PaDiM); a set of anomalous
training images gives the target signatures. Test patches are scored with the adaptive cosine
estimator (ACE) between the whitened patch embedding and the whitened signature.

Reference:
    A. Ibarra and J. Peeples, "Patch distribution modeling framework adaptive cosine estimator
    (PaDiM-ACE) for anomaly detection and localization in synthetic aperture radar imagery",
    SPIE Algorithms for Synthetic Aperture Radar Imagery XXXII, 2025 (arXiv:2504.08049).

Example:
    >>> from SARIAD.datasets import SSDD
    >>> from SARIAD.models import PadimACE
    >>> from anomalib.engine import Engine

    >>> # anomalous images used *only* to define the target signatures. Keep them disjoint from
    >>> # the images the model is tested on.
    >>> model = PadimACE(signature_dir="datasets/SSDD/anomalous_train_images")
    >>> Engine().fit(model=model, datamodule=SSDD())
"""

import logging
from pathlib import Path

import torch
from torch import nn
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms.v2.functional import to_dtype

from anomalib import LearningType
from anomalib.metrics import Evaluator
from anomalib.models.image.padim import Padim
from anomalib.post_processing import PostProcessor
from anomalib.visualization import Visualizer

from .torch_model import PadimACEModel

logger = logging.getLogger(__name__)

__all__ = ["PadimACE"]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class PadimACE(Padim):
    """PaDiM with adaptive cosine estimator scoring.

    Args:
        backbone (str, optional): timm backbone. Defaults to ``"resnet18"``.
        layers (tuple[str, ...], optional): Backbone layers to use.
        pre_trained (bool, optional): Use pretrained weights. Defaults to ``True``.
        n_features (int | None, optional): Embedding dimensions kept. ``None`` uses PaDiM's default.
        scoring (str, optional): ``"ace"`` or ``"mahalanobis"`` (plain PaDiM). Defaults to ``"ace"``.
        cov_type (str, optional): ``"full"``, ``"diagonal"`` or ``"isotropic"`` background covariance.
        whitening (str, optional): ``"reference"`` (matches the original PaDiM-ACE code) or ``"standard"``
            (textbook ACE). See :class:`~.torch_model.PadimACEModel`.
        signature_dir (str | Path | None, optional): Directory (searched recursively) with the anomalous
            training images that define the target signatures. Required to fit with ``scoring="ace"``.
        pre_processor, post_processor, evaluator, visualizer: As for any anomalib model.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: tuple[str, ...] = ("layer1", "layer2", "layer3"),
        pre_trained: bool = True,
        n_features: int | None = None,
        scoring: str = "ace",
        cov_type: str = "full",
        whitening: str = "reference",
        signature_dir: str | Path | None = None,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        # Skip Padim.__init__ (it would build a plain PadimModel) but run the base initializers.
        super(Padim, self).__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )
        self.model: PadimACEModel = PadimACEModel(
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            n_features=n_features,
            scoring=scoring,
            cov_type=cov_type,
            whitening=whitening,
        )
        self.signature_dir = Path(signature_dir) if signature_dir is not None else None

    def _anomalous_batches(self, batch_size: int = 16):
        """Anomalous training images, pre-processed like the images the model sees."""
        paths = sorted(p for p in self.signature_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            msg = f"No images found in signature_dir={self.signature_dir}"
            raise FileNotFoundError(msg)
        transform = self.pre_processor.transform
        batch: list[torch.Tensor] = []
        for path in paths:
            image = to_dtype(decode_image(str(path), mode=ImageReadMode.RGB), torch.float32, scale=True)
            batch.append(transform(tv_tensors.Image(image[None])).as_subclass(torch.Tensor))  # one-image batch
            if len(batch) == batch_size:
                yield torch.cat(batch).to(self.device)
                batch = []
        if batch:
            yield torch.cat(batch).to(self.device)

    def fit(self) -> None:
        """Fit the background Gaussian, then the target signatures (ACE scoring)."""
        super().fit()
        if self.model.scoring != "ace":
            return
        if self.signature_dir is None:
            if self.model.signatures.numel() == 0:
                msg = "PaDiM-ACE needs anomalous training images to define its target signatures: pass signature_dir=..."
                raise ValueError(msg)
            return
        self.model.fit_signatures(self._anomalous_batches())

    @property
    def learning_type(self) -> LearningType:
        """Always one-class, although ACE also uses a few anomalous images for its signatures.

        Do not report FEW_SHOT: anomalib's Engine then skips training and only validates,
        but the background statistics are collected in the training pass.
        """
        return LearningType.ONE_CLASS

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        return PostProcessor()
