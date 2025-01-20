import unittest
from pathlib import Path

import torch

from src.oracle_mnist.data import OracleMNISTDummy, OracleMNISTModuleDummy


class TestData(unittest.TestCase):
    def setUp(self):
        self.raw_data_path = Path("data/raw")
        self.processed_data_path = Path("data/processed/basic_28/train")
        self.batch_size = 32

    def test_dataset_loading(self):
        dataset = OracleMNISTDummy(
            data_paths=list(self.processed_data_path.glob("**/*.npy")), use_rgb=True, data_shape=(3, 28, 28), num_datapoints=1
        )
        self.assertGreater(len(dataset), 0, "Dataset is empty.")
        data, label = dataset[0]
        self.assertEqual(data.shape, (3, 28, 28), "Data shape is incorrect.")
        self.assertEqual(data.dtype, torch.float32, "Data type is incorrect.")
        self.assertTrue(0 <= label < 10, "Label is out of bounds.")

    def test_dataloader(self):
        data_module = OracleMNISTModuleDummy(
            batch_size=self.batch_size, in_memory_dataset=False
        )
        data_module.prepare_data()
        data_module.setup("fit")
        train_loader = data_module.train_dataloader()
        batch = next(iter(train_loader))
        x, y = batch
        self.assertEqual(
            x.shape, (self.batch_size, 3, 28, 28), "Batch shape is incorrect."
        )
        self.assertEqual(y.shape, (self.batch_size,), "Label batch shape is incorrect.")
        self.assertTrue(
            torch.is_tensor(x) and torch.is_tensor(y), "Batch is not a tensor."
        )


if __name__ == "__main__":
    unittest.main()
