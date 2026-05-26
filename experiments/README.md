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

Use the output folder under `runs/` to compare results across runs. For each experiment, record hypothesis, protocol, and results (metrics + visualizations).
