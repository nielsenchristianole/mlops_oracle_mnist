import os
from pathlib import Path

import hydra
import torch
from dotenv import load_dotenv,find_dotenv
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger, WandbLogger
from omegaconf import DictConfig

import wandb

# Import the data loading class
from oracle_mnist.modules.train_module import MNISTModule

PROJECT_NAME = "oracle_mnist"
load_dotenv(".env")


@hydra.main(config_path="/gcs/cloud_mlops_bucket/configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    seed_everything(cfg.misc.seed)
    torch.set_float32_matmul_precision(cfg.misc.precision)

    data_module = hydra.utils.instantiate(cfg.data_loader)
    criterion = torch.nn.functional.cross_entropy

    train_module = MNISTModule(
        timm_model_kwargs=cfg.model,
        optimizer_kwargs=cfg.train.optimizer,
        lr_scheduler_kwargs=cfg.train.scheduler,
        criterion=criterion,
    )

    model_input_shape = (
        1,
        3 if cfg.data_loader.use_rgb else 1,
        cfg.data_loader.imsize,
        cfg.data_loader.imsize,
    )

    model_checkpoint = ModelCheckpoint(filename="best", monitor="val_acc", save_top_k=1, save_last=True, mode="max")
    callbacks = [model_checkpoint]

    # Initialize wandb

    logger: Logger | None = None

    if cfg.misc.wandb_logging:
        wandb_api_key = os.environ.get("WANDB_API_KEY", None)
        print("foo")
        if wandb_api_key is None:
            print("WANDB_API_KEY not found in environment variables")
            
        wandb.login(key=wandb_api_key)

    # Use Lightning Trainer
    trainer = Trainer(
        max_epochs=cfg.train.epochs,
        callbacks=callbacks,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=logger,
        default_root_dir="/gcs/cloud_mlops_bucket/",
    )
    trainer.fit(train_module, data_module)

    # test
    trainer.test(ckpt_path=model_checkpoint.best_model_path, model=train_module, datamodule=data_module)

    # Export to onnx
    export_model = train_module.__class__.load_from_checkpoint(
        model_checkpoint.best_model_path
    ).eval()  # TODO replace with better way to load model
    export_model.to_onnx(
        file_path=Path(model_checkpoint.best_model_path).with_suffix(".onnx"),
        input_sample=torch.randn(*model_input_shape),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )


if __name__ == "__main__":
    train()
