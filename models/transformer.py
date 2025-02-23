# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 13:14:31 2025
Transformer for anomaly detection
@author: jpeeples
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

class Transformer(pl.LightningModule):
    def __init__(self, input_dim=256, num_layers=4, num_heads=8, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, input_dim, input_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # Add positional encoding
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch_size, input_dim)

        # Pass through transformer
        x = self.encoder(x)

        # Classification head
        x = x.mean(dim=0)
        return torch.sigmoid(self.fc(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.BCELoss()(y_hat, y.float())
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)