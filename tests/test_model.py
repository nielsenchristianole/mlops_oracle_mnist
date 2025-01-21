import unittest

import hydra
import torch
from lightning import Trainer

from src.oracle_mnist.modules.train_module import MNISTModule


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize Hydra and load the configuration
        with hydra.initialize(config_path="../configs", version_base=None):
            cls.config = hydra.compose(config_name="config")

        # Manually correct the _target_ paths for the data module and scheduler
        cls.config.data_loader._target_ = "src.oracle_mnist.data.OracleMNISTModuleDummy"
        cls.config.train.scheduler._target_ = "src.oracle_mnist.scheduler.sarphiv_scheduler.get_schedular"

        # Instantiate the data module
        # remove imsize, as its not needed to dummy data loader
        del cls.config.data_loader.imsize
        print(cls.config.data_loader)
        cls.data_module = hydra.utils.instantiate(cls.config.data_loader)
        cls.data_module.prepare_data()
        cls.data_module.setup("fit")

        # Create the training module
        cls.train_module = MNISTModule(
            timm_model_kwargs=cls.config.model,
            optimizer_kwargs=cls.config.train.optimizer,
            lr_scheduler_kwargs=cls.config.train.scheduler,
            criterion=torch.nn.functional.cross_entropy,
        )

    def test_training_one_epoch(self):
        # Run the trainer for one epoch
        trainer = Trainer(
            max_epochs=1,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            logger=False,  # Disable logging for tests
            enable_checkpointing=False,  # Disable checkpointing for tests
        )
        trainer.fit(self.train_module, datamodule=self.data_module)

        # Assert that training runs successfully and logs are generated
        logs = trainer.logged_metrics
        self.assertIn("train_loss", logs, "Training logs do not contain 'train_loss'.")
        self.assertIn("train_acc", logs, "Training logs do not contain 'train_acc'.")
        self.assertGreaterEqual(logs["train_acc"], 0, "Train accuracy is less than 0.")

    def test_model_structure(self):
        # Verify the model is initialized correctly
        self.assertEqual(
            self.train_module.model.num_classes,
            self.config.model.num_classes,
            "Model's num_classes does not match the configuration.",
        )
        self.assertTrue(hasattr(self.train_module.model, "forward"), "Model does not have a forward method.")


if __name__ == "__main__":
    unittest.main()
