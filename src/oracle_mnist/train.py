import torch
import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from oracle_mnist.modules.train_module import MNISTModule
# from data import load_oracle_mnist_data  # Import the data loading function
from data import DummyDataset
import timm
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

# @app.command()
@hydra.main(config_path = "../../configs", config_name = "config")
def train(cfg : DictConfig) -> None:
    
    seed_everything(cfg.misc.seed)
    
    # Define paths and parameters
    data_dir = cfg.data.processed_dir
    batch_size = cfg.train.batch_size

    data = DummyDataset()
    train_set,  validation_set = data.load_data()
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_data_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    # Initialize the model
    model = timm.create_model(**cfg.model)
    optimizer = hydra.utils.instantiate(cfg.train.optimizer, model.parameters())
    scheduler = hydra.utils.call(cfg.train.scheduler, optimizer = optimizer)
    criterion = nn.CrossEntropyLoss()

    train_module = MNISTModule(model, optimizer, scheduler, criterion)    

    # Use Lightning Trainer
    trainer = Trainer(max_epochs=cfg.train.epochs,
                      accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(train_module, train_dataloader, validation_data_loader)
    

if __name__ == "__main__":
    train()
    # app()
