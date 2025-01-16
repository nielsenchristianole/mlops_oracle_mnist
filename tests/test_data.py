import pytorch_lightning as pl
from oracle_mnist.data import OracleMNISTModuleBasic


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = OracleMNISTModuleBasic("data/raw")
    assert isinstance(dataset, pl.LightningDataModule)
