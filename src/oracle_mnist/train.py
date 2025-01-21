from pathlib import Path
import torch
import torch.nn as nn
from omegaconf import DictConfig
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint

# Import the data loading class
from oracle_mnist.modules.train_module import MNISTModule
from lightning.pytorch.loggers import WandbLogger, Logger, TensorBoardLogger
from oracle_mnist.data import OracleMNISTModuleBasic  # Import the data loading class

import timm
import argparse
import yaml

import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv
import wandb
import os

PROJECT_NAME = "oracle_mnist"
load_dotenv() # Load the .env file

@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    seed_everything(cfg.misc.seed)
    torch.set_float32_matmul_precision(cfg.misc.precision)
    

    data_module = hydra.utils.instantiate(cfg.data_loader)
    criterion = torch.nn.functional.cross_entropy

    train_module = MNISTModule(
        timm_model_kwargs=cfg.model,
        optimizer_kwargs=cfg.train.optimizer,
        lr_scheduler_kwargs=cfg.train.scheduler,
        criterion=criterion)

    model_input_shape = (
        1,
        3 if cfg.data_loader.use_rgb else 1,
        cfg.data_loader.imsize,
        cfg.data_loader.imsize)

    model_checkpoint = ModelCheckpoint(
        filename="best", monitor="val_acc", save_top_k=1, save_last=True, mode="max"
    )
    callbacks = [model_checkpoint]
    
    ### Initialize wandb
    
    logger : Logger|None = None
    
    if cfg.misc.wandb_logging:
        if wandb_api_key := os.getenv("WANDB_API_KEY"):
            wandb.login(key=wandb_api_key)
        else:
            print("No API key found in environment variables. Logging in manually:")
            wandb.login()
        
        logger = WandbLogger(project=PROJECT_NAME)

    # Use Lightning Trainer
    trainer = Trainer(max_epochs=cfg.train.epochs,
                    callbacks=callbacks,
                    accelerator="gpu" if torch.cuda.is_available() else "cpu",
                    logger=logger)
        
    trainer.fit(train_module, data_module)

    # test
    trainer.test(
        ckpt_path=model_checkpoint.best_model_path,
        model=train_module,
        datamodule=data_module)
    
    # Export to onnx
    export_model = train_module.__class__.load_from_checkpoint(model_checkpoint.best_model_path).eval() # TODO replace with better way to load model
    export_model.to_onnx(
        file_path=Path(model_checkpoint.best_model_path).with_suffix('.onnx'),
        input_sample=torch.randn(*model_input_shape),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}})


def sweep_train():
    with wandb.init() as run:
        # Import and initialize Hydra configuration
        with hydra.initialize(config_path="../../configs", version_base=None):
            cfg = hydra.compose(config_name="config")

        config = wandb.config

        # Override Hydra configurations with sweep parameters
        cfg.train.optimizer.lr = config.learning_rate
        cfg.train.batch_size = config.batch_size
        cfg.train.epochs = config.epochs

        # Create and run the training module using the updated configurations
        data_module = hydra.utils.instantiate(cfg.data_loader)
        train_module = MNISTModule(
            timm_model_kwargs=cfg.model,
            optimizer_kwargs=cfg.train.optimizer,
            lr_scheduler_kwargs=cfg.train.scheduler,
            criterion=torch.nn.functional.cross_entropy
        )

        trainer = Trainer(
            max_epochs=cfg.train.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=WandbLogger(project=PROJECT_NAME, save_dir="outputs/sweeps"),  # Redirect WandB logs
            default_root_dir="outputs/sweeps"  # Specify custom directory for sweeps
        )
        trainer.fit(train_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run a WandB sweep")
    parser.add_argument("--sweep_count", type=int, default=10, help="Number of runs for the sweep")
    args = parser.parse_args()

    if args.sweep:
        # Load sweep configuration
        with open("configs/sweep_config.yaml", "r") as f:
            sweep_config = yaml.safe_load(f)

        # Initialize and run the sweep
        sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
        wandb.agent(sweep_id, function=sweep_train, count=args.sweep_count)
    else:
        train()
