import numpy as np

from pydantic import BaseModel
from numpydantic import NDArray, Shape


class Image(BaseModel):
    array: NDArray[Shape['* b, 3 c, 28 y, 28 x'], int]


class Confidence(BaseModel):
    array: NDArray[Shape['* b, 10 c'], float]


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=1, keepdims=True)
