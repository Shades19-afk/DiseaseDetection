# Disease Detection — Research-oriented ML Engineering

This repository evolves the DenseNet pneumonia detection baseline into a research-focused CV project. Goals:
- Improve model explainability (Grad-CAM)
- Add robust evaluation metrics and analysis (F1, ROC-AUC, confusion matrix)
- Experiment with augmentations and class-imbalance handling
- Keep DenseNet as the primary baseline and add model-comparison experiments
- Provide a clean, modular code structure and TensorBoard logging

See `experiments/README.md` for suggested experiments and rationale.

## Baseline experiment

Run the first end-to-end DenseNet baseline with automatic class weight handling, TensorBoard logging, checkpointing, evaluation metrics, Grad-CAM, and false-positive/false-negative inspection:

```bash
python run_baseline.py --data-dir path/to/data --batch-size 16 --epochs 10
```

This pipeline executes these research-friendly stages:
- **Dataset loading:** builds a `tf.data` pipeline from image folders and resizes/normalizes inputs.
- **Class imbalance handling:** computes class weights from the training split so the model does not ignore rare cases.
- **Baseline model:** uses DenseNet121 as the primary feature extractor.
- **Training:** logs loss/accuracy to TensorBoard and saves the best checkpoint.
- **Evaluation:** computes precision, recall, F1, ROC-AUC, and confusion matrix.
- **Explainability and error analysis:** creates Grad-CAM visualizations and saves false positives/false negatives for manual review.

The script creates a timestamped experiment folder under `runs/` with:
- TensorBoard logs in `tensorboard/`
- model checkpoint in `checkpoints/best_model.h5`
- evaluation results in `metrics.json`
- confusion matrix plot in `confusion_matrix.png`
- Grad-CAM outputs in `gradcam/`
- misclassified images in `misclassified/

## Augmentation ablation harness

After running the baseline pipeline, compare augmentation strength with:

```bash
python run_augmentation_ablation.py --data-dir path/to/data --batch-size 16 --epochs 10
```

This harness automatically runs three experiments: `none`, `light`, and `strong` augmentation. It saves separate metrics, confusion matrices, TensorBoard logs, Grad-CAM outputs, and a final `augmentation_comparison.csv` summary.
