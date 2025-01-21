import numpy as np
def normalize_data(data: np.ndarray) -> np.ndarray:
    # Normalizing the intensities of the image to [0, 1].
    return (data / 255.0).astype(np.float32)