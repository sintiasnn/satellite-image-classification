import numpy as np
from PIL import Image

from src.core.config import settings


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((settings.model_input_size, settings.model_input_size))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array
