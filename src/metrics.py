import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, dataset):
    y_true = []
    y_pred = []
    y_scores = []
    binary = None

    for x,y in dataset:
        preds = model.predict(x)
        if preds.shape[-1] == 1:
            probs = preds.ravel()
            pred_labels = (probs > 0.5).astype(int)
            y_scores.extend(probs.tolist())
            binary = True
        else:
            pred_labels = preds.argmax(axis=1)
            if preds.shape[-1] == 2:
                y_scores.extend(preds[:, 1].tolist())
            else:
                y_scores.extend(preds.tolist())
            binary = len(preds.shape) == 2 and preds.shape[-1] == 2

        y_true.extend(y.numpy().ravel().tolist())
        y_pred.extend(pred_labels.ravel().tolist())

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        if binary:
            auc = roc_auc_score(y_true, y_scores)
        else:
            auc = None
    except Exception:
        auc = None
    cm = confusion_matrix(y_true, y_pred)
    return dict(precision=precision, recall=recall, f1=f1, roc_auc=auc, confusion_matrix=cm)

def plot_confusion(cm, labels=None):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
