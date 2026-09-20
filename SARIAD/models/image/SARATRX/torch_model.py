"""PyTorch model for SARATRX.

SARATRX wraps the HiViT masked autoencoder (MAE) of SARATR-X (Li et al., 2024), pretrained
on a large collection of SAR imagery. Only the *encoder* is released, so the decoder starts
from random weights and has to be trained on normal images (``SARATRX.fit``); the encoder can
either be finetuned too or frozen (``freeze_encoder=True``, "training just the head").

The decoder predicts three multi-scale SAR gradient features for the masked patches.
At inference, the error between those predictions and the true features (see
:mod:`.anomaly_map`) is the anomaly signal.

Example:
    >>> from SARIAD.models.image.SARATRX.torch_model import SARATRXModel
    >>> model = SARATRXModel().eval()
    >>> out = model(torch.rand(4, 1, 224, 224))     # InferenceBatch(pred_score, anomaly_map)
"""

import logging

import torch
from torch import nn

from anomalib.data import InferenceBatch
from anomalib.utils.path import get_pretrained_weights_dir
from SARIAD.models.image.SARATRX.SARATRX.pretraining.models.models_hivit_mae import mae_hivit_base_dec512d6b
from SARIAD.utils.blob_utils import fetch_blob

from .anomaly_map import AnomalyMapGenerator

logger = logging.getLogger(__name__)

CHECKPOINT_NAME = "mae_hivit_base_1600ep.pth"
CHECKPOINT_DRIVE_ID = "1VZQz4buhlepZ5akTcEvrA3a_nxsQZ8eQ"
IMAGE_SIZE = 224
NUM_SCALES = 3  # multi-scale SAR gradient features predicted by the decoder

# Keys the pretrained checkpoint legitimately does not contain: the decoder and its fixed
# positional embedding (trained here), and the fixed feature-extraction kernels (built in code).
_NOT_IN_CHECKPOINT = ("decoder_", "sarfeature")


class SARATRXModel(nn.Module):
    """HiViT-MAE anomaly detector for SAR images.

    Args:
        checkpoint_path (str | None, optional): Pretrained encoder weights. ``None`` downloads
            the SARATR-X checkpoint into anomalib's pretrained-weights cache.
        mask_ratio (float, optional): Fraction of patches masked, both when training and when
            scoring. Defaults to ``0.75``.
        num_mask_passes (int, optional): Number of random masks used to score an image. Every
            patch's error is averaged over the passes in which it was masked. Defaults to ``4``.
        sigma (int, optional): Std. dev. of the Gaussian smoothing of the anomaly map. Defaults to ``4``.
        error_mode (str, optional): ``"l2"`` or ``"l1"`` feature error. Defaults to ``"l2"``.
        freeze_encoder (bool, optional): Only train the decoder. Needs far less memory.
            Defaults to ``False``.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        mask_ratio: float = 0.75,
        num_mask_passes: int = 4,
        sigma: int = 4,
        error_mode: str = "l2",
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            msg = f"mask_ratio must be in (0, 1), got {mask_ratio}"
            raise ValueError(msg)

        self.mask_ratio = mask_ratio
        self.num_mask_passes = num_mask_passes
        # This is the configuration the released checkpoint was trained with (no relative
        # position bias, 6 decoder blocks); other settings leave encoder weights uninitialized.
        self.model = mae_hivit_base_dec512d6b()
        self._load_encoder_weights(checkpoint_path)
        self.anomaly_map_generator = AnomalyMapGenerator(sigma=sigma, error_mode=error_mode)

        if freeze_encoder:
            for name, param in self.model.named_parameters():
                if not name.startswith(("decoder_", "mask_token")):
                    param.requires_grad = False

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info("SARATRX trainable parameters: %.2fM", n_trainable / 1e6)

    def _load_encoder_weights(self, checkpoint_path: str | None) -> None:
        if checkpoint_path is None:
            path = get_pretrained_weights_dir() / CHECKPOINT_NAME
            fetch_blob(str(path), drive_file_id=CHECKPOINT_DRIVE_ID, is_archive=False)
        else:
            path = checkpoint_path

        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = state_dict.get("model", state_dict)
        result = self.model.load_state_dict(state_dict, strict=False)

        missing = [k for k in result.missing_keys if not k.startswith(_NOT_IN_CHECKPOINT)]
        if missing:
            msg = f"Checkpoint {path} is missing {len(missing)} encoder weights (e.g. {missing[:3]})"
            raise RuntimeError(msg)
        if result.unexpected_keys:
            logger.warning("Ignoring %d unexpected checkpoint keys, e.g. %s", len(result.unexpected_keys), result.unexpected_keys[:3])

    def _prepare(self, images: torch.Tensor) -> torch.Tensor:
        """Validate the input and reduce it to the single channel the MAE expects."""
        if images.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
            msg = f"SARATRX expects {IMAGE_SIZE}x{IMAGE_SIZE} inputs, got {tuple(images.shape[-2:])}. Resize them in the pre-processor."
            raise ValueError(msg)
        return images.mean(dim=1, keepdim=True) if images.shape[1] != 1 else images

    @torch.no_grad()
    def sar_features(self, images: torch.Tensor) -> torch.Tensor:
        """Target features ``(B, L, 3 * p * p)``, exactly as the MAE's training loss builds them."""
        mae = self.model
        target = torch.cat([mae.patchify(getattr(mae, f"sarfeature{i}")(images)) for i in range(1, NUM_SCALES + 1)], dim=-1)
        if mae.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
        return target

    def forward(self, images: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Score a ``(B, C, 224, 224)`` batch.

        Returns:
            The MAE feature-reconstruction loss when training, otherwise an ``InferenceBatch``
            with the image-level ``pred_score`` (max of the map) and the pixel-level ``anomaly_map``.
        """
        images = self._prepare(images)
        if self.training:
            loss, _, _ = self.model(images, self.mask_ratio)
            return loss

        with torch.no_grad():
            target = self.sar_features(images)
            error_sum, mask_count = 0.0, 0.0
            for _ in range(self.num_mask_passes):
                _, pred, mask = self.model(images, self.mask_ratio)  # mask: 1 = masked (predicted)
                error_sum = error_sum + self.anomaly_map_generator.patch_error(pred, target, NUM_SCALES) * mask.unsqueeze(-1)
                mask_count = mask_count + mask
            anomaly_map = self.anomaly_map_generator(error_sum, mask_count, IMAGE_SIZE, self.model.patch_embed.patch_size[0])
            pred_score = torch.amax(anomaly_map, dim=(-2, -1)).reshape(-1)
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
