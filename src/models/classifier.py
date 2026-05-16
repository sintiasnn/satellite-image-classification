import tensorflow as tf
import numpy as np

from src.core.config import settings


class SatelliteClassifier:
    def __init__(self):
        self.model = None
        self.class_names = settings.model_classes

    def load_model(self):
        self.model = tf.keras.models.load_model(settings.model_path)

    def predict(self, image_array: np.ndarray) -> tuple[str, dict[str, float]]:
        predictions = self.model.predict(image_array, verbose=0)[0]
        predicted_class_idx = int(np.argmax(predictions))
        predicted_class = self.class_names[predicted_class_idx]
        confidence = {
            name: float(score) for name, score in zip(self.class_names, predictions)
        }
        return predicted_class, confidence


classifier = SatelliteClassifier()
