import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from src.explainability import make_gradcam_heatmap, overlay_heatmap


@dataclass(frozen=True)
class PredictionResult:
    label: str
    confidence: float
    raw_probabilities: np.ndarray
    heatmap: np.ndarray
    overlay: np.ndarray
    input_size: Tuple[int, int]


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / 'model.best4.keras'
DEFAULT_CLASS_NAMES = ("Normal", "Pneumonia")


def load_model(model_path: str | os.PathLike | None = None) -> tf.keras.Model:
    resolved_path = Path(model_path or os.getenv('MODEL_PATH', DEFAULT_MODEL_PATH))
    if not resolved_path.exists():
        raise FileNotFoundError(f'Model file not found: {resolved_path}')
    model = tf.keras.models.load_model(str(resolved_path), compile=False)
    return model


def get_input_size(model: tf.keras.Model) -> Tuple[int, int]:
    shape = model.input_shape
    if len(shape) >= 4:
        return int(shape[1]), int(shape[2])
    raise ValueError('Model input shape is not a 4D tensor.')


def _collect_layers(layer):
    layers = []
    if hasattr(layer, 'layers'):
        for sublayer in layer.layers:
            layers.extend(_collect_layers(sublayer))
    layers.append(layer)
    return layers


def find_gradcam_target_layer(model: tf.keras.Model) -> str:
    all_layers = []
    for layer in model.layers:
        all_layers.extend(_collect_layers(layer))
    for layer in reversed(all_layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError('No Conv2D layer found in the model graph.')


def preprocess_image(uploaded_image, target_size: Tuple[int, int]) -> np.ndarray:
    image = ImageOps.exif_transpose(Image.open(uploaded_image).convert('RGB'))
    image = image.resize(target_size)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return image_array


def predict_image(uploaded_image, model: tf.keras.Model, class_names: Tuple[str, ...] = DEFAULT_CLASS_NAMES) -> PredictionResult:
    input_size = get_input_size(model)
    image_array = preprocess_image(uploaded_image, input_size)
    batch = np.expand_dims(image_array, axis=0)
    probabilities = model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_label = class_names[predicted_index] if predicted_index < len(class_names) else f'Class {predicted_index}'
    confidence = float(np.max(probabilities))

    target_layer_name = find_gradcam_target_layer(model)
    heatmap = make_gradcam_heatmap(batch, model, target_layer_name)
    overlay = overlay_heatmap(image_array, heatmap, alpha=0.4)

    return PredictionResult(
        label=predicted_label,
        confidence=confidence,
        raw_probabilities=probabilities,
        heatmap=heatmap,
        overlay=overlay,
        input_size=input_size,
    )
