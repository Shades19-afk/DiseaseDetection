import numpy as np
import tensorflow as tf


def _iter_layers(layer):
    yield layer
    if hasattr(layer, 'layers'):
        for sublayer in layer.layers:
            yield from _iter_layers(sublayer)


def resolve_layer(model, layer_name: str):
    direct_layer = None
    try:
        direct_layer = model.get_layer(layer_name)
    except ValueError:
        direct_layer = None

    if direct_layer is not None:
        return direct_layer

    for layer in _iter_layers(model):
        if layer.name == layer_name:
            return layer

    raise ValueError(f'Layer "{layer_name}" not found in model graph.')


def find_parent_model(model, layer_name: str):
    for layer in model.layers:
        if hasattr(layer, 'layers'):
            if any(sub_layer.name == layer_name for sub_layer in layer.layers):
                return layer
            nested_parent = find_parent_model(layer, layer_name)
            if nested_parent is not None:
                return nested_parent
    return None


def build_gradcam_model(model, last_conv_layer_name):
    parent_model = find_parent_model(model, last_conv_layer_name)
    if parent_model is not None:
        base_input = parent_model.input
        conv_outputs = parent_model.get_layer(last_conv_layer_name).output
        x = parent_model(base_input)
        for layer in model.layers:
            if layer is parent_model:
                continue
            x = layer(x)
        return tf.keras.models.Model(inputs=base_input, outputs=[conv_outputs, x])

    x = model.input
    conv_outputs = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_outputs = x
    return tf.keras.models.Model(inputs=model.input, outputs=[conv_outputs, x])


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    parent_model = find_parent_model(model, last_conv_layer_name)
    if parent_model is not None:
        target_layer = resolve_layer(parent_model, last_conv_layer_name)
        conv_model = tf.keras.models.Model(inputs=parent_model.input, outputs=target_layer.output)
    else:
        target_layer = resolve_layer(model, last_conv_layer_name)
        conv_model = tf.keras.models.Model(inputs=model.input, outputs=target_layer.output)

    conv_outputs = conv_model(img_array, training=False)
    predictions = model(img_array, training=False)
    if pred_index is None:
        pred_index = int(tf.argmax(predictions[0]).numpy()) if predictions.shape[-1] > 1 else 0

    heatmap = None
    try:
        with tf.GradientTape() as tape:
            conv_outputs_tape = conv_model(img_array, training=False)
            predictions_tape = model(img_array, training=False)
            class_channel = predictions_tape[:, pred_index]
            grads = tape.gradient(class_channel, conv_outputs_tape)
        if grads is not None:
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            heatmap = tf.reduce_sum(conv_outputs_tape[0] * pooled_grads, axis=-1)
    except Exception:
        heatmap = None

    if heatmap is None:
        heatmap = tf.reduce_mean(conv_outputs[0], axis=-1)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4):
    import cv2

    img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_uint8, 1.0 - alpha, heatmap_colored, alpha, 0)
    return overlay.astype(np.uint8)
