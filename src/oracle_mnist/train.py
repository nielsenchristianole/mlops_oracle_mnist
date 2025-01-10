import os
import torch
from torch.utils.data import DataLoader
from lightning import Trainer
from model import MNISTModel
from data import load_oracle_mnist_data  # Import the data loading function

if __name__ == "__main__":
    # Define paths and parameters
    data_dir = "data/processed"  # Adjust this path if necessary
    batch_size = 64
    num_classes = 10  # Oracle-MNIST has 10 classes
    learning_rate = 0.001
    num_epochs = 5

    # Load the data using the data.py function
    train_loader, val_loader = load_oracle_mnist_data(data_dir, batch_size)

    # Initialize the model
    model = MNISTModel(num_classes=num_classes, learning_rate=learning_rate)

    # Use Lightning Trainer
    trainer = Trainer(max_epochs=num_epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu")
    trainer.fit(model, train_loader, val_loader)
