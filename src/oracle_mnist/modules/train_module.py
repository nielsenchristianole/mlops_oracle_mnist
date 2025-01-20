from lightning import LightningModule
from typing import Callable
import torch
import timm
import hydra

class MNISTModule(LightningModule):

    def __init__(
        self,
        timm_model_kwargs: dict,
        optimizer_kwargs: dict,
        lr_scheduler_kwargs: dict,
        criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model: torch.nn.Module = timm.create_model(**timm_model_kwargs)
        self.optimizer = hydra.utils.instantiate(optimizer_kwargs, self.model.parameters())
        self.lr_scheduler = hydra.utils.call(lr_scheduler_kwargs, optimizer=self.optimizer)
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def handle_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.handle_step(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.handle_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.handle_step(batch, batch_idx)
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss

    def configure_optimizers(self):
        return {  # type: ignore
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "lr"
            }
        }
