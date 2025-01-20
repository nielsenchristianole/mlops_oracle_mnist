from lightning import LightningModule
import torch



class MNISTModule(LightningModule):

    def __init__(self, model: torch.nn.Module, optimizer, lr_scheduler, criterion):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
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
