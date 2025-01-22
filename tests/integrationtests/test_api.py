import unittest

import numpy as np
import requests

from oracle_mnist.utils.api_utils import Image


class TestAPI(unittest.TestCase):

    @classmethod
    def test_batch(cls):
        cls._test_batch(1)
        cls._test_batch(2)
    
    @staticmethod
    def _test_batch(batch_size: int):

        response = requests.post("http://0.0.0.0:6060/predict", json={"im": np.random.randint(0, 255, (batch_size, 3, 28, 28), dtype=np.uint8).tolist()})
        assert response.status_code == 200, f"{response.status_code=}\n\n{response.text=}"
        conf = np.array(response.json())
        assert conf.shape == (batch_size, 10), f"{conf.shape=}"

    @staticmethod
    def test_health():
        response = requests.get("http://0.0.0.0:6060/healthz")
        assert response.status_code == 200, f"{response.status_code=}\n\n{response.text=}"



if __name__ == "__main__":
    unittest.main()