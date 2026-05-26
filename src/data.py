import tensorflow as tf
from typing import Tuple

def build_dataset(data_dir: str, img_size: Tuple[int,int]=(224,224), batch_size: int=32, augment: bool=False, subset: str=None, validation_split: float=0.2, seed: int=123):
    base_args = dict(image_size=img_size, batch_size=batch_size, seed=seed)
    if subset in ("training","validation"):
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='int',
            subset=subset,
            validation_split=validation_split,
            shuffle=True,
            **base_args,
        )
    else:
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            data_dir,
            labels='inferred',
            label_mode='int',
            shuffle=True,
            **base_args,
        )

    normalization = tf.keras.layers.Rescaling(1./255)
    resize = tf.keras.layers.Resizing(img_size[0], img_size[1])

    def prep(x,y):
        x = resize(x)
        x = normalization(x)
        return x,y

    ds = ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.08),
        ])

        def apply_aug(x,y):
            return aug(x, training=True), y
        ds = ds.map(apply_aug, num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)
