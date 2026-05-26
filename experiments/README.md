# Experiments and Rationale

This document lists suggested incremental experiments to make the project research-oriented.

1. Augmentation ablation: compare no-aug, light TF preprocessing, and stronger geometric/color augmentations.
2. Class-imbalance handling: compare class weights, oversampling minority, and focal loss.
3. Model comparison: DenseNet121 vs ResNet50 vs EfficientNetB0 (same training protocol).
4. Explainability: run Grad-CAM on random FP/FN to understand failure modes.
5. Logging: use TensorBoard to monitor train/val metrics and sample Grad-CAMs.

Baseline experiment command:

```bash
python run_baseline.py --data-dir path/to/image-data --batch-size 16 --epochs 10
```

Augmentation ablation command:

```bash
python run_augmentation_ablation.py --data-dir path/to/image-data --batch-size 16 --epochs 10
```

This will run three experimental conditions (`none`, `light`, `strong`) using the same train/validation split. Each condition saves separate metrics, confusion matrices, Grad-CAM outputs, and a summary table.

## Architecture comparison

Run a structured architecture comparison with:

```bash
python run_model_comparison.py --data-dir path/to/image-data --batch-size 16 --epochs 10 --augmentation light
```

This compares:
- DenseNet121
- ResNet50
- EfficientNetB0

All models use the same preprocessing pipeline and train/validation split. The output includes per-model metrics, TensorBoard logs, confusion matrices, Grad-CAM examples, and a final `architecture_comparison.csv`.

### Notes on model tradeoffs

- **DenseNet121**: good parameter efficiency and feature reuse, often strong on medical image tasks where fine-grained texture matters.
- **ResNet50**: robust residual learning, good baseline for transfer learning and slightly heavier than DenseNet.
- **EfficientNetB0**: smallest model here, useful when inference cost matters and deployment efficiency is a priority.

Use the final summary table to decide whether accuracy gains justify extra model size or training time.

Use the output folder under `runs/` to compare results across runs. For each experiment, record hypothesis, protocol, and results (metrics + visualizations).
