import hydra
import torch
import yaml
from dotenv import load_dotenv
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger

import wandb
from oracle_mnist.modules.train_module import MNISTModule

PROJECT_NAME = "oracle_mnist"
load_dotenv()  # Load the .env file


def sweep_train():
    with wandb.init() as _:
        # Import and initialize Hydra configuration
        with hydra.initialize(config_path="../../configs", version_base=None):
            cfg = hydra.compose(config_name="config")

        config = wandb.config

        # Override Hydra configurations with sweep parameters
        cfg.train.optimizer.lr = config.learning_rate
        cfg.train.batch_size = config.batch_size
        cfg.model.model_name = config.model_name

        # Create and run the training module using the updated configurations
        data_module = hydra.utils.instantiate(cfg.data_loader)
        train_module = MNISTModule(
            timm_model_kwargs=cfg.model,
            optimizer_kwargs=cfg.train.optimizer,
            lr_scheduler_kwargs=cfg.train.scheduler,
            criterion=torch.nn.functional.cross_entropy,
        )

        trainer = Trainer(
            max_epochs=cfg.train.epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=WandbLogger(project=PROJECT_NAME, save_dir="outputs/sweeps"),  # Redirect WandB logs
            default_root_dir="outputs/sweeps",  # Specify custom directory for sweeps
        )
        trainer.fit(train_module, data_module)


if __name__ == "__main__":
    # Load sweep configuration
    with open("configs/sweep_config.yaml", "r") as f:
        sweep_config = yaml.safe_load(f)

    # Initialize and run the sweep
    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=sweep_train)
