import hydra
import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

# Import the data loading class
from oracle_mnist.modules.train_module import MNISTModule


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    seed_everything(cfg.misc.seed)

    # Define paths and parameters
    # data_dir = cfg.data.processed_dir
    # batch_size = cfg.train.batch_size

    # Initialize the model
    model = timm.create_model(**cfg.model)

    data_module = hydra.utils.instantiate(cfg.data_loader)
    optimizer = hydra.utils.instantiate(cfg.train.optimizer, model.parameters())
    scheduler = hydra.utils.call(cfg.train.scheduler, optimizer=optimizer)
    criterion = nn.CrossEntropyLoss()

    train_module = MNISTModule(model, optimizer, scheduler, criterion)

    model_checkpoint = ModelCheckpoint(
        filename="best", monitor="val_acc", save_top_k=1, save_last=True, mode="max"
    )
    callbacks = [model_checkpoint]

    # Use Lightning Trainer
    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        callbacks=callbacks,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )
    trainer.fit(train_module, data_module)

    # test
    trainer.test(
        ckpt_path=model_checkpoint.best_model_path,
        model=train_module,
        datamodule=data_module,
    )

    # TODO: export to onnx
    # train_module.load_from_checkpoint(model_checkpoint.best_model_path)


if __name__ == "__main__":
    train()
