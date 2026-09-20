"""SARATRX: SAR anomaly detection with the SARATR-X hierarchical vision transformer.

A HiViT masked autoencoder, pretrained on SAR imagery (SARATR-X), is trained on *normal*
images to predict multi-scale SAR gradient features of masked patches. Anomalies are the
regions whose features it cannot predict.

The released checkpoint only contains the encoder, so the decoder always has to be trained
(``engine.fit``). Use ``freeze_encoder=True`` to train only the decoder.

Example:
    >>> from SARIAD.datasets import MSTAR
    >>> from SARIAD.models import SARATRX
    >>> from anomalib.engine import Engine

    >>> datamodule = MSTAR()
    >>> model = SARATRX(freeze_encoder=True)
    >>> engine = Engine(max_epochs=20)
    >>> engine.fit(model=model, datamodule=datamodule)
    >>> predictions = engine.predict(model=model, datamodule=datamodule)
"""

import contextlib
import io
import logging

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchvision.transforms.v2 import Compose, Grayscale, Resize

from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.pre_processing import PreProcessor
from anomalib.visualization import Visualizer
from SARIAD.models.image.SARATRX.SARATRX.pretraining.util.lr_decay import param_groups_lrd

from .torch_model import IMAGE_SIZE, SARATRXModel

logger = logging.getLogger(__name__)

__all__ = ["SARATRX"]


class SARATRX(AnomalibModule):
    """SARATRX: SAR anomaly detection with a HiViT masked autoencoder.

    Args:
        checkpoint_path (str | None, optional): Pretrained encoder checkpoint. ``None``
            downloads the SARATR-X checkpoint. Defaults to ``None``.
        mask_ratio (float, optional): Fraction of patches masked. Defaults to ``0.75``.
        num_mask_passes (int, optional): Random masks used to score an image. Defaults to ``4``.
        freeze_encoder (bool, optional): Only train the decoder. Defaults to ``False``.
        learning_rate (float, optional): Learning rate of the top layers; lower layers are scaled
            down by ``layer_decay``. Defaults to ``2.5e-4``.
        weight_decay (float, optional): AdamW weight decay. Defaults to ``0.05``.
        layer_decay (float, optional): Layer-wise learning rate decay factor. Defaults to ``0.75``.
        pre_processor (PreProcessor | bool, optional): ``True`` uses :meth:`configure_pre_processor`
            (resize to 224 and grayscale). Defaults to ``True``.
        post_processor (PostProcessor | bool, optional): Defaults to ``True``.
        evaluator (Evaluator | bool, optional): Defaults to ``True``.
        visualizer (Visualizer | bool, optional): Defaults to ``True``.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        mask_ratio: float = 0.75,
        num_mask_passes: int = 4,
        freeze_encoder: bool = False,
        learning_rate: float = 2.5e-4,
        weight_decay: float = 0.05,
        layer_decay: float = 0.75,
        pre_processor: nn.Module | bool = True,
        post_processor: nn.Module | bool = True,
        evaluator: Evaluator | bool = True,
        visualizer: Visualizer | bool = True,
    ) -> None:
        super().__init__(
            pre_processor=pre_processor,
            post_processor=post_processor,
            evaluator=evaluator,
            visualizer=visualizer,
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.layer_decay = layer_decay

        self.model = SARATRXModel(
            checkpoint_path=checkpoint_path,
            mask_ratio=mask_ratio,
            num_mask_passes=num_mask_passes,
            freeze_encoder=freeze_encoder,
        )

    @staticmethod
    def configure_pre_processor(image_size: tuple[int, int] | None = None) -> PreProcessor:
        """Resize to the 224x224 the model was pretrained with and convert to grayscale.

        The grayscale image keeps 3 (identical) channels because anomalib's ``Batch`` only
        accepts 3-channel images; the model reduces it to the single channel it expects.
        ``image_size`` is accepted for API compatibility but the model only supports 224x224.
        """
        return PreProcessor(Compose([Resize((IMAGE_SIZE, IMAGE_SIZE)), Grayscale(num_output_channels=3)]))

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """AdamW with layer-wise learning rate decay (from the SARATR-X finetuning recipe)."""
        # param_groups_lrd prints every group; keep the logs readable
        with contextlib.redirect_stdout(io.StringIO()):
            param_groups = param_groups_lrd(
                self.model.model,
                self.weight_decay,
                no_weight_decay_list=self.model.model.no_weight_decay(),
                layer_decay=self.layer_decay,
            )
        for group in param_groups:
            group["lr"] = self.learning_rate * group.get("lr_scale", 1.0)
        return torch.optim.AdamW(param_groups, lr=self.learning_rate, betas=(0.9, 0.999))

    def training_step(self, batch: Batch, batch_idx: int) -> STEP_OUTPUT:
        """Masked feature-reconstruction loss on a batch of normal images."""
        del batch_idx
        loss = self.model(batch.image)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch.image))
        return {"loss": loss}

    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        """Predict the anomaly score and map of a batch."""
        del args, kwargs
        predictions = self.model(batch.image)
        return batch.update(**predictions._asdict())

    @property
    def trainer_arguments(self) -> dict[str, int | float | list]:
        """Default trainer arguments (the accelerator is left to Lightning's auto detection)."""
        return {"max_epochs": 20, "check_val_every_n_epoch": 5}

    @property
    def learning_type(self) -> LearningType:
        """Trained on normal images only."""
        return LearningType.ONE_CLASS
