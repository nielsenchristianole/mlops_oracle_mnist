import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel

Image = NDArray[Shape["* b, 3 c, 28 y, 28 x"], np.uint8]
Confidence = NDArray[Shape["* b, 10 l"], np.float32]
Prediction = NDArray[Shape["* b"], np.int32]


# package confidence and prediction
class Result(BaseModel):
    confidence: Confidence
    prediction: Prediction
