from ultralytics import YOLO
import torch
from torch import nn
from torch.nn import functional as F
from anomalib.data import InferenceBatch
from anomalib.models.components import MultiVariateGaussian 

class YOLOAnomalyModel(nn.Module):
    def __init__(self, model_path = "yolov8n.pt"):
        super().__init__()
        self.yolo = YOLO(model_path)

    def forward(self, input_tensor):
