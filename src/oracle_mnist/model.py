import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from lightning import LightningModule

class MNISTModel(LightningModule):
    def __init__(self, num_classes: int = 10, learning_rate: float = 0.001):
        super().__init__()
        self.model = timm.create_model("mobilenetv3_small_100", pretrained=True, num_classes=num_classes)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
