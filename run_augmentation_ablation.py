import argparse
import json
from datetime import datetime
from pathlib import Path

from PIL import Image
import pandas as pd
import tensorflow as tf

from src.data import build_dataset
from src.error_analysis import save_misclassified
from src.explainability import make_gradcam_heatmap, overlay_heatmap
from src.metrics import evaluate_model, plot_confusion
from src.models import get_densenet
from src.train import train_model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run augmentation ablation experiments for a DenseNet baseline.'
    )
    parser.add_argument('--data-dir', type=str, required=True, help='Root directory containing class subfolders.')
    parser.add_argument('--val-dir', type=str, default=None, help='Optional separate validation directory path.')
    parser.add_argument('--img-size', type=int, default=224, help='Square image size for training and evaluation.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for tf.data pipeline.')
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Adam learning rate.')
    parser.add_argument('--output-dir', type=str, default='runs', help='Base folder for experiment output.')
    parser.add_argument('--experiment-name', type=str, default='augmentation_ablation', help='Name for this experiment run.')
    parser.add_argument('--augmentations', type=str, default='none,light,strong', help='Comma-separated augmentation strength presets to compare.')
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


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


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


def build_datasets(args, augmentation: str):
    if args.val_dir:
        train_ds = build_dataset(
            args.data_dir,
            img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            augmentation=augmentation,
            subset=None,
        )
        val_ds = build_dataset(
            args.val_dir,
            img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            augment=False,
            subset=None,
        )
    else:
        train_ds = build_dataset(
            args.data_dir,
            img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            augmentation=augmentation,
            subset='training',
            validation_split=0.2,
            seed=123,
        )
        val_ds = build_dataset(
            args.data_dir,
            img_size=(args.img_size, args.img_size),
            batch_size=args.batch_size,
            augment=False,
            subset='validation',
            validation_split=0.2,
            seed=123,
        )

    return train_ds, val_ds


def run_experiment(args, augmentation: str, global_out: Path):
    run_name = f'{augmentation}_aug'
    run_dir = global_out / run_name
    tensorboard_dir = run_dir / 'tensorboard'
    checkpoint_path = run_dir / 'checkpoints' / 'best_model.h5'
    metrics_path = run_dir / 'metrics.json'
    confusion_path = run_dir / 'confusion_matrix.png'
    gradcam_dir = run_dir / 'gradcam'
    misclassified_dir = run_dir / 'misclassified'

    print(f'=== Running augmentation experiment: {augmentation} ===')
    train_ds, val_ds = build_datasets(args, augmentation)
    print(f'Classes: {train_ds.class_names}')

    class_weights = compute_class_weights(train_ds)
    print(f'Class weights: {class_weights}')

    num_classes = len(train_ds.class_names)
    model = get_densenet(input_shape=(args.img_size, args.img_size, 3), num_classes=1 if num_classes == 2 else num_classes)

    train_model(
        model,
        train_ds,
        val_ds,
        epochs=args.epochs,
        lr=args.learning_rate,
        class_weight=class_weights,
        log_dir=str(tensorboard_dir),
        checkpoint_path=str(checkpoint_path),
    )

    metrics = evaluate_model(model, val_ds)
    save_json(metrics, metrics_path)

    import matplotlib.pyplot as plt
    plot_confusion(metrics['confusion_matrix'], labels=train_ds.class_names)
    confusion_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(confusion_path, dpi=200)
    plt.close()

    save_misclassified(model, val_ds, str(misclassified_dir))
    last_conv_layer = find_last_conv_layer(model)
    save_gradcam_examples(model, val_ds, gradcam_dir, last_conv_layer_name=last_conv_layer, max_examples=3)

    summary = {
        'augmentation': augmentation,
        'metrics': metrics,
        'checkpoint': str(checkpoint_path),
        'tensorboard_dir': str(tensorboard_dir),
        'gradcam_dir': str(gradcam_dir),
        'misclassified_dir': str(misclassified_dir),
    }
    save_json(summary, run_dir / 'summary.json')
    return summary


def save_comparison_table(summaries, output_dir: Path):
    records = []
    for item in summaries:
        row = {
            'augmentation': item['augmentation'],
            'precision': item['metrics']['precision'],
            'recall': item['metrics']['recall'],
            'f1': item['metrics']['f1'],
            'roc_auc': item['metrics']['roc_auc'],
        }
        records.append(row)

    df = pd.DataFrame(records)
    df = df.sort_values('augmentation')
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'augmentation_comparison.csv'
    df.to_csv(csv_path, index=False)
    return df


def main():
    args = parse_args()
    global_out = Path(args.output_dir) / f'{args.experiment_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    global_out.mkdir(parents=True, exist_ok=True)

    augmentations = [x.strip() for x in args.augmentations.split(',') if x.strip()]
    summaries = []
    for augmentation in augmentations:
        summary = run_experiment(args, augmentation, global_out)
        summaries.append(summary)

    comparison_df = save_comparison_table(summaries, global_out)
    save_json({'comparison': comparison_df.to_dict(orient='records')}, global_out / 'comparison_summary.json')

    print('\nFinal augmentation comparison:')
    print(comparison_df.to_string(index=False))
    print(f'All run artifacts and summaries saved under {global_out}')
    print('Review TensorBoard logs for each run under each subfolder.')


if __name__ == '__main__':
    main()
