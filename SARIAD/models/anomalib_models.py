# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:10:07 2025
Call anomalib models for SAR images
@author: jpeeples
"""

import importlib
import pytorch_lightning as pl

class AnomalibModel(pl.LightningModule):
    def __init__(self, model_name: str, input_size=(256, 256), **kwargs):
        super().__init__()
        self.model_name = model_name
        self.input_size = input_size

        try:
            # Dynamically import the specified model from anomalib.models
            module_name = f"anomalib.models.{model_name}"
            module = importlib.import_module(module_name)
            model_class = getattr(module, model_name.capitalize())  # Class name is usually capitalized
            self.model = model_class(input_size=input_size, **kwargs)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Model '{model_name}' not found in Anomalib: {e}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self.model.training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.model.validation_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self.model.test_step(batch, batch_idx)

    def configure_optimizers(self):
        return self.model.configure_optimizers()