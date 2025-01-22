import numpy as np
import onnxruntime as rt
import bentoml
from fastapi import FastAPI, Depends

from oracle_mnist.utils.api_utils import softmax, Image, Confidence
from oracle_mnist.utils.data_utils import normalize_data


MODEL_PATH = '../../models/model.onnx'
PROVIDER_LIST = ('CPUExecutionProvider', )


app = FastAPI()


@bentoml.service(
    workers=1)
@bentoml.asgi_app(app, path="/v1")
class ImageClassifierService:
    """Image classifier service using ONNX model."""

    name = "image-classifier"

    def __init__(self) -> None:
        self.ort_session = rt.InferenceSession(MODEL_PATH, providers=PROVIDER_LIST)

    @bentoml.api(
        batchable=True,
        batch_dim=(0, 0),
        # input_spec=Image,
        # output_spec=Confidence,
        max_batch_size=128,
        max_latency_ms=1000)
    def predict(self, im: list[list[list[list[int]]]]) -> list[list[float]]:
        """Predict the class of the input image."""
        im = normalize_data(np.array(im))
        model_output, = self.ort_session.run(None, {'input': im})
        return softmax(model_output).tolist()


@app.get("/service_name")
async def hello(service: ImageClassifierService = Depends(bentoml.get_current_service)):
    # Outside service class, use `Depends` to get the service
    return f"{service.name}"


@app.get("/hello_{name}")
async def hello(name: str):
    # Outside service class, use `Depends` to get the service
    return f"Hello {name}"
