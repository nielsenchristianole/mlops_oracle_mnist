from pathlib import Path

import typer
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset

class DummyDataset(Dataset):
    """Dummy dataset for testing purposes."""

    def __init__(self, length: int = 100) -> None:
        self.length = length

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return {"index": index}
    
    def load_data():
        # Generate dummy data
        data = np.random.randint(0, 256, (1000, 28, 28), dtype=np.uint8)
        # Generate random targets
        targets = np.random.randint(0, 10, 1000)
        # Normalize the data
        data = data / 255.0
        # Split the data and targets into train and test sets
        train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.2, random_state=42)

        # Convert to PyTorch tensors
        train_tensors = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)
        test_tensors = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)
        train_targets = torch.tensor(train_targets, dtype=torch.long)
        test_targets = torch.tensor(test_targets, dtype=torch.long)

        # Create train and test TensorDatasets
        train_set = TensorDataset(train_tensors, train_targets)
        test_set = TensorDataset(test_tensors, test_targets)

        return train_set, test_set

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, raw_data_path: Path) -> None:
        self.data_path = raw_data_path

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""

def preprocess(raw_data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(raw_data_path)
    dataset.preprocess(output_folder)


if __name__ == "__main__":
    typer.run(preprocess)
