import tensorflow as tf
from typing import Optional
import os

def train_model(model: tf.keras.Model, train_ds, val_ds, epochs: int=10, lr: float=1e-4, class_weight: Optional[dict]=None, log_dir: Optional[str]=None, checkpoint_path: Optional[str]=None):
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = 'binary_crossentropy' if model.output_shape[-1]==1 else 'categorical_crossentropy'
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    callbacks = []
    if log_dir:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir))
    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss'))

    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight, callbacks=callbacks)
    return history
