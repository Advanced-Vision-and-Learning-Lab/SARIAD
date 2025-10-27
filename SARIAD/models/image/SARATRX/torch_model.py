"""PyTorch model for the SARATRX model implementation.

This module implements the SARATRX model architecture using PyTorch. SARATRX uses
a HiViT-based Masked Autoencoder (MAE) for anomaly detection by learning to
reconstruct normal patterns from grayscale images.

The model extracts high-level features from a hierarchical vision transformer
backbone trained with masked autoencoding. During inference, reconstruction
errors indicate anomalies.

Example:
    >>> from SARIAD.models.image.SARATRX.torch_model import SaratrxModel
    >>> model = SaratrxModel()
    >>> input_tensor = torch.randn(32, 1, 224, 224)
    >>> output = model(input_tensor)
"""

import logging
import torch
from torch import nn
from torch.nn import functional as F

from anomalib.data import InferenceBatch
from SARIAD.models.image.SARATRX.SARATRX.pretraining.models.models_hivit_mae import HiViTMaskedAutoencoder
from SARIAD.models.image.SARATRX.SARATRX.pretraining.util.pos_embed import interpolate_pos_embed
from SARIAD.utils.blob_utils import fetch_blob

from .anomaly_map import AnomalyMapGenerator

logger = logging.getLogger(__name__)


class SARATRXModel(nn.Module):
    """SARATRX Model using HiViT Masked Autoencoder.

    This model uses a hierarchical vision transformer with masked autoencoding
    for anomaly detection. It reconstructs masked portions of input images and
    uses reconstruction error as an anomaly signal.

    Args:
        checkpoint_path (str, optional): Path to pretrained checkpoint. If None,
            downloads default checkpoint. Defaults to None.
        mask_ratio (float, optional): Ratio of patches to mask during inference.
            Defaults to 0.75.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        mask_ratio: float = 0.75,
    ) -> None:
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.model = HiViTMaskedAutoencoder(hifeat=True)
        
        # Load pretrained weights
        if checkpoint_path is None:
            logger.info("Downloading pretrained HiViT-MAE checkpoint...")
            fetch_blob(
                "mae_hivit_base_1600ep.pth",
                drive_file_id="1VZQz4buhlepZ5akTcEvrA3a_nxsQZ8eQ",
                is_archive=False
            )
            checkpoint_path = "mae_hivit_base_1600ep.pth"
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = self.model.state_dict()
        
        # Remove incompatible keys
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
                logger.info(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        
        # Interpolate positional embeddings if needed
        interpolate_pos_embed(self.model, checkpoint)
        
        # Load weights
        msg = self.model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Loaded pretrained weights: {msg}")
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f'Number of trainable parameters: {n_parameters / 1.e6:.2f}M')
        
        self.anomaly_map_generator = AnomalyMapGenerator()
        
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor | InferenceBatch:
        """Forward-pass image-batch (N, C, H, W) through the model.

        Args:
            input_tensor (torch.Tensor): Image batch with shape (N, C, H, W)
                Expected to be grayscale (C=1) with size 224x224.

        Returns:
            torch.Tensor | InferenceBatch: If training, returns the loss tensor.
                If inference, returns ``InferenceBatch`` containing prediction
                scores and anomaly maps.
        """
        if self.training:
            # During training, return loss
            loss, _, _ = self.model(input_tensor)
            return loss
        
        # During inference, generate anomaly map
        with torch.no_grad():
            loss, pred, mask = self.model(input_tensor)
            
            # Generate anomaly map from reconstruction
            anomaly_map = self.anomaly_map_generator(
                input_image=input_tensor,
                reconstruction=pred,
                mask=mask,
                patch_size=self.model.patch_embed.patch_size[0]
            )
            
            # Compute anomaly score as maximum value in anomaly map
            pred_score = torch.amax(anomaly_map, dim=(-2, -1))
            
        return InferenceBatch(pred_score=pred_score, anomaly_map=anomaly_map)
    
    @staticmethod
    def unpatch(x: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
        """Convert flat patches back to 2D image format.
        
        Args:
            x (torch.Tensor): Flattened patches tensor
            patch_size (int): Size of each patch. Defaults to 16.
            
        Returns:
            torch.Tensor: Reconstructed 2D image tensor
        """
        B = x.shape[0]
        h = w = 14  # For 224x224 image with patch_size=16
        chans = 1   # Grayscale
        
        # Reshape flat patches into 2D image
        x = x.reshape(B, h, w, patch_size, patch_size, chans)
        
        # Rearrange to standard image format (B, C, H, W)
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(B, chans, h * patch_size, w * patch_size)
        
        return x
