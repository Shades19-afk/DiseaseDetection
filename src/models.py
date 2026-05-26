import tensorflow as tf
from typing import Callable, Dict, Tuple


def get_densenet(input_shape: Tuple[int,int,3]=(224,224,3), num_classes: int=1, weights: str='imagenet'):
    base = tf.keras.applications.DenseNet121(include_top=False, weights=weights, input_shape=input_shape)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    if num_classes == 1:
        out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    else:
        out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=base.input, outputs=out)


def get_resnet(input_shape: Tuple[int,int,3]=(224,224,3), num_classes: int=1, weights: str='imagenet'):
    base = tf.keras.applications.ResNet50(include_top=False, weights=weights, input_shape=input_shape)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    if num_classes == 1:
        out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    else:
        out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=base.input, outputs=out)


def get_efficientnet(input_shape: Tuple[int,int,3]=(224,224,3), num_classes: int=1, weights: str='imagenet'):
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights=weights, input_shape=input_shape)
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.3)(x)
    if num_classes == 1:
        out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    else:
        out = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=base.input, outputs=out)


def get_model_by_name(name: str, input_shape: Tuple[int,int,3]=(224,224,3), num_classes: int=1, weights: str='imagenet') -> tf.keras.Model:
    architecture_map: Dict[str, Callable[..., tf.keras.Model]] = {
        'densenet': get_densenet,
        'resnet': get_resnet,
        'efficientnet': get_efficientnet,
    }
    key = name.lower()
    if key not in architecture_map:
        raise ValueError(f'Unsupported model name: {name}. Valid options are {list(architecture_map.keys())}.')
    return architecture_map[key](input_shape=input_shape, num_classes=num_classes, weights=weights)
