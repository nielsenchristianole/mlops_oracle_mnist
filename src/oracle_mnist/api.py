import numpy as np
import onnxruntime as rt
import bentoml

from oracle_mnist.utils.api_utils import Confidence, Image, softmax
from oracle_mnist.utils.data_utils import normalize_data


MODEL_PATH = '../../models/model.onnx'
PROVIDER_LIST = ('CPUExecutionProvider', )


@bentoml.service(
    workers=1)
class ImageClassifierService:
    """Image classifier service using ONNX model."""

    def __init__(self) -> None:
        self.ort_session = rt.InferenceSession(MODEL_PATH, providers=PROVIDER_LIST)

    @bentoml.api(
        batchable=True,
        batch_dim=(0, 0),
        max_batch_size=128,
        max_latency_ms=1000)
    def predict(self, im: list) -> list:
        """Predict the class of the input image."""
        im = normalize_data(np.array(im))
        model_output, = self.ort_session.run(None, {'input': im})
        return softmax(model_output).tolist()
