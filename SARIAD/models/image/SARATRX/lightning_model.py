"""SARATRX: SAR Anomaly Detection using Hierarchical Vision Transformer.

This model implements the SARATRX algorithm for anomaly detection using a
Hierarchical Vision Transformer (HiViT) with Masked Autoencoding (MAE).

The model learns to reconstruct normal patterns from grayscale SAR images.
During inference, reconstruction errors indicate anomalies.

Example:
    >>> from anomalib.data import MVTecAD
    >>> from SARIAD.models.image.SARATRX import SARATRX
    >>> from anomalib.engine import Engine

    >>> # Initialize model and data
    >>> datamodule = MVTecAD()
    >>> model = SARATRX()

    >>> # Train using the Engine
    >>> engine = Engine()
    >>> engine.fit(model=model, datamodule=datamodule)

    >>> # Get predictions
    >>> predictions = engine.predict(model=model, datamodule=datamodule)
"""

import logging
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchvision.transforms.v2 import Compose, Resize, Grayscale

from anomalib.data import Batch
from anomalib.models.components import AnomalibModule
from anomalib import LearningType
from anomalib.post_processing import PostProcessor
from anomalib.pre_processing import PreProcessor
from anomalib.metrics import Evaluator
from anomalib.visualization import Visualizer
from SARIAD.models.image.SARATRX.SARATRX.pretraining.util.lr_decay import param_groups_lrd
from SARIAD.models.image.SARATRX.SARATRX.pretraining.util.misc import NativeScalerWithGradNormCount as NativeScaler

from .torch_model import SARATRXModel

logger = logging.getLogger(__name__)

__all__ = ["SARATRX"]


class SARATRX(AnomalibModule):
    """SARATRX: SAR Anomaly Detection using HiViT-MAE.

    This model uses a Hierarchical Vision Transformer with Masked Autoencoding
    for anomaly detection in SAR imagery. It learns to reconstruct masked
    portions of normal images, and uses reconstruction error as an anomaly signal.

    Args:
        checkpoint_path (str | None, optional): Path to pretrained HiViT-MAE
            checkpoint. If None, downloads default checkpoint. Defaults to None.
        mask_ratio (float, optional): Ratio of patches to mask during training.
            Defaults to 0.75.
        learning_rate (float, optional): Base learning rate. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay for optimizer.
            Defaults to 0.05.
        layer_decay (float, optional): Layer-wise learning rate decay factor.
            Defaults to 0.75.
        pre_processor (PreProcessor | bool, optional): Preprocessor to apply on
            input data. Defaults to True.
        post_processor (PostProcessor | bool, optional): Post processor to apply
            on model outputs. Defaults to True.
        evaluator (Evaluator | bool, optional): Evaluator for computing metrics.
            Defaults to True.
        visualizer (Visualizer | bool, optional): Visualizer for generating
            result images. Defaults to True.

    Example:
        >>> from SARIAD.models.image.SARATRX import SARATRX
        >>> model = SARATRX(learning_rate=1e-3, mask_ratio=0.75)
        >>> # Model is ready for training with Engine
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        mask_ratio: float = 0.75,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.05,
        layer_decay: float = 0.75,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        # Setup preprocessing transform
        if pre_processor is True:
            transform = Compose([
                Resize((224, 224)),
                Grayscale(num_output_channels=1),
            ])
            pre_processor = PreProcessor(transform)
        
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        # Model hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.layer_decay = layer_decay
        self.mask_ratio = mask_ratio
        
        # Initialize model
        self.model: SARATRXModel = SARATRXModel(
            checkpoint_path=checkpoint_path,
            mask_ratio=mask_ratio,
        )
        
        # Manual optimization for custom training loop with loss scaler
        self.automatic_optimization = False
        self.loss_scaler = NativeScaler()
        
        logger.info(f"SARATRX model initialized with learning_rate={learning_rate}, "
                   f"mask_ratio={mask_ratio}")

    def configure_optimizers(self) -> torch.optim.Optimizer | None:
        """Configure optimizer with layer-wise learning rate decay.

        Returns:
            torch.optim.Optimizer | None: AdamW optimizer with layer-wise LR decay,
                or None if using manual optimization.
        """
        if not self.automatic_optimization:
            # With manual optimization, we create optimizer but don't return it
            # Build optimizer with layer-wise lr decay (lrd)
            param_groups = param_groups_lrd(
                self.model.model,  # Access the inner MAE model
                self.weight_decay,
                no_weight_decay_list=self.model.model.no_weight_decay(),
                layer_decay=self.layer_decay
            )
            
            # Calculate effective learning rate
            eff_batch_size = 64  # Adjust based on your batch size
            lr = self.learning_rate * eff_batch_size / 256
            
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=lr,
                betas=(0.9, 0.999)
            )
            logger.info(f"Optimizer configured with base_lr={self.learning_rate:.2e}, "
                       f"actual_lr={lr:.2e}")
            return None
        
        # For automatic optimization (not used with current settings)
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        """Perform the training step with MAE reconstruction loss.

        Args:
            batch (Batch): Input batch containing image and metadata
            batch_idx (int): Index of the current batch

        Returns:
            STEP_OUTPUT: Dictionary containing the loss value
        """
        optimizer = self.optimizer
        optimizer.zero_grad()

        # Forward pass with automatic mixed precision
        with torch.cuda.amp.autocast():
            loss = self.model(batch.image)

        # Backward pass with gradient scaling
        self.loss_scaler(
            loss,
            optimizer,
            clip_grad=None,
            parameters=self.model.parameters(),
        )

        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True, 
                prog_bar=True, logger=True)

        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Perform a validation step.

        During validation, the model generates predictions including anomaly
        scores and maps.

        Args:
            batch (Batch): Input batch containing image and metadata
            args: Additional arguments (unused)
            kwargs: Additional keyword arguments (unused)

        Returns:
            STEP_OUTPUT: Updated batch with predictions
        """
        del args, kwargs  # These variables are not used.

        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, int | float | list]:
        """Get default trainer arguments for SARATRX.

        Returns:
            dict[str, Any]: Trainer arguments including:
                - accelerator: GPU acceleration
                - devices: Number of GPUs (1 for manual optimization)
                - max_epochs: Number of training epochs
                - check_val_every_n_epoch: Validation frequency
        """
        return {
            "accelerator": "gpu",
            "devices": 1,  # Manual optimization requires single device
            "max_epochs": 20,
            "check_val_every_n_epoch": 20,
            "gradient_clip_val": None,  # Handled by loss_scaler
        }

    @property
    def learning_type(self) -> LearningType:
        """Return the learning type of the model.

        Returns:
            LearningType: Learning type (ONE_CLASS for SARATRX)
        """
        return LearningType.FEW_SHOT

    @staticmethod
    def configure_post_processor() -> PostProcessor:
        """Return the default post-processor for SARATRX.

        Returns:
            PostProcessor: Default post-processor
        """
        return PostProcessor()
