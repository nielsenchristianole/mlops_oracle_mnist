import unittest
import numpy as np
from pathlib import Path
from src.oracle_mnist.data import OracleMNISTModuleBasic

class TestProcessedData(unittest.TestCase):
    def setUp(self):
        # Set up paths for dummy processed data
        self.processed_dir = Path("./data/processed/basic_28/train")
        self.module = OracleMNISTModuleBasic(imsize=28, batch_size=1)

    def test_data_dimensions(self):
        # Prepare dummy data (make sure your processed data exists at the path)
        self.module.prepare_data()
        self.module.setup(stage="fit")
        train_loader = self.module.train_dataloader()

        # Test data dimensions for one batch
        for x, y in train_loader:
            self.assertEqual(x.shape, (1, 3, 28, 28))  # Batch size x Channels x Height x Width
            self.assertEqual(y.shape, (1,))  # Batch size
            break

if __name__ == "__main__":
    unittest.main()
