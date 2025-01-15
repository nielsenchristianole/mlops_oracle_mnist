import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from oracle_mnist.modules.train_module import MNISTModule
from oracle_mnist.data import OracleMNISTModuleBasic  # Import the data loading class

import timm

import hydra
from omegaconf import DictConfig

# @app.command()
@hydra.main(config_path = "../../configs", config_name = "config", version_base=None)
def train(cfg : DictConfig) -> None:
    
    seed_everything(cfg.misc.seed)
    
    # Define paths and parameters
    # data_dir = cfg.data.processed_dir
    # batch_size = cfg.train.batch_size

    # Initialize the model
    model = timm.create_model(**cfg.model)
    
    data_module = hydra.utils.instantiate(cfg.data_loader)
    optimizer = hydra.utils.instantiate(cfg.train.optimizer, model.parameters())
    scheduler = hydra.utils.call(cfg.train.scheduler, optimizer = optimizer)
    criterion = nn.CrossEntropyLoss()

    train_module = MNISTModule(model, optimizer, scheduler, criterion)    

    # Use Lightning Trainer
    trainer = Trainer(max_epochs=cfg.train.epochs,
                      accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(train_module, data_module)
    

if __name__ == "__main__":
    train()
    # app()
