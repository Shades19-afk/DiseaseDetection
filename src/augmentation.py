import tensorflow as tf
from typing import Optional


def get_augmentation_pipeline(level: Optional[str] = None):
    if level is None or level.lower() == 'none':
        return None

    level = level.lower()
    if level == 'light':
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.05),
        ])

    if level == 'strong':
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.12),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomContrast(0.15),
        ])

    raise ValueError(f'Unknown augmentation level: {level}. Use none, light, or strong.')
