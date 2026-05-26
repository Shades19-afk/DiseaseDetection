import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

from src.data import build_dataset
from src.error_analysis import save_misclassified
from src.explainability import make_gradcam_heatmap, overlay_heatmap
from src.metrics import evaluate_model, plot_confusion
from src.models import get_densenet
from src.train import train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a reproducible DenseNet baseline experiment for medical image detection.'
    )
    parser.add_argument('--data-dir', type=str, required=True, help='Root directory containing class subfolders.')
    parser.add_argument('--val-dir', type=str, default=None, help='Optional separate validation directory path.')
    parser.add_argument('--img-size', type=int, default=224, help='Square image size for training and evaluation.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for tf.data pipeline.')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Adam learning rate.')
    parser.add_argument('--output-dir', type=str, default='runs', help='Base folder for experiment output.')
    parser.add_argument('--experiment-name', type=str, default='baseline_densenet', help='Name for this experiment run.')
    return parser.parse_args()


def compute_class_weights(dataset):
    counts = {}
    for _, labels in dataset:
        labels = labels.numpy().ravel().tolist()
        for label in labels:
            counts[label] = counts.get(label, 0) + 1

    total = sum(counts.values())
    num_classes = len(counts)
    return {label: total / (num_classes * count) for label, count in counts.items()}


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    raise ValueError('No Conv2D layer found in model. Ensure the model is a convolutional architecture.')


def save_metrics(metrics, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)


def save_gradcam_examples(model, dataset, output_dir: Path, last_conv_layer_name: str, max_examples: int = 3):
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_saved = 0
    for batch in dataset:
        images, labels = batch
        preds = model.predict(images)
        if preds.shape[-1] == 1:
            probs = preds.ravel()
            pred_labels = (probs > 0.5).astype(int)
        else:
            probs = preds[:, 1] if preds.shape[-1] > 1 else preds[:, 0]
            pred_labels = preds.argmax(axis=1)

        for i in range(len(images)):
            if examples_saved >= max_examples:
                return

            img = images[i].numpy()
            heatmap = make_gradcam_heatmap(tf.expand_dims(img, axis=0), model, last_conv_layer_name)
            overlay = overlay_heatmap(img, heatmap)
            filename = output_dir / f'gradcam_{examples_saved}_true{int(labels[i].numpy())}_pred{int(pred_labels[i])}.png'
            Image.fromarray(overlay).save(filename)
            examples_saved += 1


def main():
    args = parse_args()
    run_name = f'{args.experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    out_dir = Path(args.output_dir) / run_name
    tensorboard_dir = out_dir / 'tensorboard'
    checkpoint_path = out_dir / 'checkpoints' / 'best_model.h5'
    metrics_path = out_dir / 'metrics.json'
    confusion_path = out_dir / 'confusion_matrix.png'
    gradcam_dir = out_dir / 'gradcam'
    misclassified_dir = out_dir / 'misclassified'

    print('Stage 1: Load dataset with tf.data pipeline')
    if args.val_dir:
        train_ds = build_dataset(args.data_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size, augment=False, subset=None)
        val_ds = build_dataset(args.val_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size, augment=False, subset=None)
    else:
        train_ds = build_dataset(args.data_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size, augment=False, subset='training', validation_split=0.2)
        val_ds = build_dataset(args.data_dir, img_size=(args.img_size, args.img_size), batch_size=args.batch_size, augment=False, subset='validation', validation_split=0.2)

    print(f'Classes: {train_ds.class_names}')
    num_classes = len(train_ds.class_names)
    print(f'Number of classes: {num_classes}')

    print('Stage 2: Compute class weights to compensate for imbalance')
    class_weights = compute_class_weights(train_ds)
    print(f'Class weights: {class_weights}')

    print('Stage 3: Build DenseNet baseline model')
    model = get_densenet(input_shape=(args.img_size, args.img_size, 3), num_classes=1 if num_classes == 2 else num_classes)
    print(model.summary())

    print('Stage 4: Train model with TensorBoard logging and checkpoint saving')
    history = train_model(
        model,
        train_ds,
        val_ds,
        epochs=args.epochs,
        lr=args.learning_rate,
        class_weight=class_weights,
        log_dir=str(tensorboard_dir),
        checkpoint_path=str(checkpoint_path),
    )

    print('Stage 5: Evaluate model and compute robust metrics')
    metrics = evaluate_model(model, val_ds)
    print('Evaluation results:')
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            continue
        print(f'  {key}: {value}')
    save_metrics(metrics, metrics_path)

    print('Stage 6: Save confusion matrix plot')
    plot_confusion(metrics['confusion_matrix'], labels=train_ds.class_names)
    plt.savefig(confusion_path, dpi=200)
    plt.close()

    print('Stage 7: Save false positives and false negatives for inspection')
    save_misclassified(model, val_ds, str(misclassified_dir))

    print('Stage 8: Generate a few Grad-CAM visualizations')
    last_conv_layer = find_last_conv_layer(model)
    save_gradcam_examples(model, val_ds, gradcam_dir, last_conv_layer_name=last_conv_layer, max_examples=3)

    summary = {
        'run_name': run_name,
        'output_dir': str(out_dir),
        'tensorboard_dir': str(tensorboard_dir),
        'checkpoint': str(checkpoint_path),
        'metrics': metrics,
        'gradcam_dir': str(gradcam_dir),
        'misclassified_dir': str(misclassified_dir),
    }
    save_metrics(summary, out_dir / 'run_summary.json')
    print('Baseline experiment complete.')
    print(f'Outputs saved to {out_dir}')
    print('Start TensorBoard with: tensorboard --logdir', tensorboard_dir)


if __name__ == '__main__':
    main()
