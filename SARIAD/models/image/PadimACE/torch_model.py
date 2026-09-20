"""PyTorch model for PaDiM-ACE.

PaDiM-ACE (Ibarra & Peeples, 2025; https://github.com/Advanced-Vision-and-Learning-Lab/PaDiM-ACE,
https://arxiv.org/abs/2504.08049) keeps PaDiM's patch-wise Gaussian model of *normal* images but scores
test patches with the Adaptive Cosine Estimator (ACE) instead of the Mahalanobis distance: the
embedding of a patch and a *target signature* are whitened with the background statistics and the
score is the cosine between them. The signature of a patch position is the mean embedding of
**anomalous training images** at that position, so this variant needs a few labelled anomalies
in addition to the normal training images. Nothing is trained by gradient descent.

Built on anomalib's :class:`~anomalib.models.image.padim.torch_model.PadimModel`, which is subclassed
rather than copied so that it follows anomalib.
"""

import logging
from collections.abc import Iterable

import torch
from torch.nn import functional as F  # noqa: N812

from anomalib.data import InferenceBatch
from anomalib.models.components.base import DynamicBufferMixin
from anomalib.models.image.padim.torch_model import PadimModel

logger = logging.getLogger(__name__)

SCORINGS = ("ace", "mahalanobis")
COV_TYPES = ("full", "diagonal", "isotropic")
WHITENINGS = ("reference", "standard")

_EPS = 1e-7


class PadimACEModel(DynamicBufferMixin, PadimModel):
    """PaDiM with ACE scoring.

    Args:
        backbone: timm backbone. Defaults to ``"resnet18"``.
        layers: Backbone layers whose features are concatenated. Defaults to the first three ResNet stages.
        pre_trained: Use pretrained backbone weights. Defaults to ``True``.
        n_features: Embedding dimensions kept (random subset). ``None`` uses PaDiM's default for the backbone.
        scoring: ``"ace"`` (needs signatures, see :meth:`fit_signatures`) or ``"mahalanobis"`` (plain PaDiM).
        cov_type: Background covariance used to whiten: ``"full"``, ``"diagonal"`` or ``"isotropic"``
            (identity: the cosine is invariant to the scalar variance, so every isotropic estimate is equivalent).
        whitening: ``"reference"`` reproduces the original PaDiM-ACE code, which decomposes the *inverse*
            covariance and applies ``D^-1/2 U^T`` to it. ``"standard"`` is textbook ACE: the data are whitened
            with ``Sigma^-1/2`` and the signature is taken relative to the background mean.
            They differ (the reference transform does not whiten; see the note in the docs).
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        layers: tuple[str, ...] | list[str] = ("layer1", "layer2", "layer3"),
        pre_trained: bool = True,
        n_features: int | None = None,
        scoring: str = "ace",
        cov_type: str = "full",
        whitening: str = "reference",
    ) -> None:
        for name, value, options in (("scoring", scoring, SCORINGS), ("cov_type", cov_type, COV_TYPES), ("whitening", whitening, WHITENINGS)):
            if value not in options:
                msg = f"{name} must be one of {options}, got {value!r}"
                raise ValueError(msg)
        super().__init__(backbone=backbone, layers=list(layers), pre_trained=pre_trained, n_features=n_features)
        self.scoring = scoring
        self.cov_type = cov_type
        self.whitening = whitening
        self.register_buffer("signatures", torch.empty(0))  # (n_features, n_patches) once fitted
        self.signatures: torch.Tensor
        self._transform: torch.Tensor | None = None  # cached whitening, rebuilt after fit()

    # ------------------------------------------------------------------ fitting
    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        """Patch embeddings ``(B, n_features, h, w)`` of a batch of (pre-processed) images."""
        return self.generate_embedding(self.feature_extractor(images))

    @torch.no_grad()
    def fit_signatures(self, batches: Iterable[torch.Tensor]) -> None:
        """Set the target signatures to the mean embedding of anomalous images, per patch position.

        Args:
            batches: Iterable of pre-processed anomalous image batches ``(B, 3, H, W)``.
        """
        total, count = None, 0
        for images in batches:
            embeddings = self.embed(images).double()
            total = embeddings.sum(dim=0) if total is None else total + embeddings.sum(dim=0)
            count += embeddings.shape[0]
        if count == 0:
            msg = "No anomalous images given to compute the target signatures."
            raise ValueError(msg)
        mean = (total / count).float()  # (n, h, w)
        self.signatures = mean.reshape(mean.shape[0], -1).contiguous()
        self._transform = None
        logger.info("Target signatures computed from %d anomalous images.", count)

    def fit(self) -> None:
        """Fit the background Gaussian on the collected normal embeddings."""
        super().fit()
        self._transform = None

    # ------------------------------------------------------------------ scoring
    def _build_transform(self) -> torch.Tensor:
        """Whitening applied to ``x - mean``: ``(P, n, n)`` for full, ``(P, n)`` for diagonal, ``(0,)`` for isotropic."""
        inv_cov = self.gaussian.inv_covariance  # (P, n, n), Sigma^-1 of every patch position
        if self.cov_type == "isotropic":
            return torch.empty(0, device=inv_cov.device)

        if self.cov_type == "diagonal":
            if self.whitening == "reference":
                scale = inv_cov.diagonal(dim1=-2, dim2=-1).abs().clamp_min(_EPS).rsqrt()  # (inv Sigma_ii)^-1/2
            else:
                evals, evecs = torch.linalg.eigh(inv_cov)
                variance = (evecs**2 / evals.clamp_min(_EPS).unsqueeze(-2)).sum(-1)  # diag(Sigma) from Sigma^-1
                scale = variance.clamp_min(_EPS).rsqrt()
            return scale

        evals, evecs = torch.linalg.eigh(inv_cov)  # Sigma^-1 = U diag(evals) U^T
        evals = evals.clamp_min(_EPS)
        # standard: Sigma^-1/2 = diag(evals^+1/2) U^T ; reference: diag(evals^-1/2) U^T (see class docstring)
        diag = evals.sqrt() if self.whitening == "standard" else evals.rsqrt()
        return diag.unsqueeze(-1) * evecs.transpose(-1, -2)

    def _whiten(self, x: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, P, n)`` or ``(P, n)``; returns the same shape, whitened per patch position."""
        if self.cov_type == "isotropic":
            return x
        if self.cov_type == "diagonal":
            return x * transform
        return torch.einsum("pij,...pj->...pi", transform, x)

    def ace_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        """ACE score (cosine between whitened embedding and whitened signature), ``(B, 1, h, w)``."""
        if self.signatures.numel() == 0:
            msg = "PaDiM-ACE has no target signatures: give the model anomalous training images (fit_signatures / signature_dir)."
            raise RuntimeError(msg)
        batch, _, height, width = embeddings.shape
        if self.signatures.shape[1] != height * width:
            msg = f"Signatures were computed for {self.signatures.shape[1]} patches, got {height * width}. Use the same image size."
            raise ValueError(msg)
        if self._transform is None or self._transform.device != embeddings.device:
            self._transform = self._build_transform().to(embeddings.device)

        mean = self.gaussian.mean.to(embeddings.device)  # (n, P)
        signature = self.signatures.to(embeddings.device)  # (n, P)
        if self.whitening == "standard":
            signature = signature - mean  # signature relative to the background, like the data

        x = (embeddings.reshape(batch, embeddings.shape[1], -1) - mean).permute(0, 2, 1)  # (B, P, n)
        x_hat = F.normalize(self._whiten(x, self._transform), dim=-1)
        s_hat = F.normalize(self._whiten(signature.T, self._transform), dim=-1)  # (P, n)
        return torch.einsum("pi,bpi->bp", s_hat, x_hat).reshape(batch, 1, height, width)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        if self.training or self.scoring == "mahalanobis":
            return super().forward(input_tensor)

        output_size = input_tensor.shape[-2:]
        if self.tiler:
            input_tensor = self.tiler.tile(input_tensor)
        embeddings = self.embed(input_tensor)
        if self.tiler:
            embeddings = self.tiler.untile(embeddings)

        score_map = self.ace_scores(embeddings)
        generator = self.anomaly_map_generator
        anomaly_map = generator.smooth_anomaly_map(generator.up_sample(score_map, output_size))
        return InferenceBatch(pred_score=torch.amax(anomaly_map, dim=(-2, -1)), anomaly_map=anomaly_map)
