import logging
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from anomalib import LearningType
from anomalib.data import Batch
from anomalib.metrics import Evaluator
from anomalib.models.components import AnomalibModule
from anomalib.post_processing import PostProcessor
from anomalib.visualization import Visualizer

from .torch_model import YOLOAnomalyModel

logger = logging.getLogger(__name__)

class YOLOAnomaly(AnomalibModule):
    def __init__(self, backbone = "yolov8n.pt", pre_processor = True, post_processor = True):
        super().__init__(pre_processor=pre_processor, post_processor=post_processor)

        self.model = YOLOAnomalyModel(model_path=backbone)

    def training_step(self, batch: Batch, *args, **kwargs) -> None:
        pass
    def fit(self) -> None:
        pass
    def validation_step(self, batch: Batch, *args, **kwargs) -> STEP_OUTPUT:
        pass
