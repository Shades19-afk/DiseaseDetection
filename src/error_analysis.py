import os
import numpy as np
from pathlib import Path
from PIL import Image

def save_misclassified(model, dataset, out_dir: str, threshold: float=0.5):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fp_dir = Path(out_dir)/'false_positives'
    fn_dir = Path(out_dir)/'false_negatives'
    fp_dir.mkdir(exist_ok=True)
    fn_dir.mkdir(exist_ok=True)
    idx = 0
    for x,y in dataset:
        preds = model.predict(x)
        if preds.shape[-1]==1:
            pred_labels = (preds.ravel()>threshold).astype(int)
        else:
            pred_labels = preds.argmax(axis=1)
        true_labels = y.numpy().ravel().astype(int)
        for i in range(len(pred_labels)):
            if pred_labels[i]==1 and true_labels[i]==0:
                Image.fromarray((x[i].numpy()*255).astype('uint8')).save(fp_dir/f'fp_{idx}.png')
                idx += 1
            if pred_labels[i]==0 and true_labels[i]==1:
                Image.fromarray((x[i].numpy()*255).astype('uint8')).save(fn_dir/f'fn_{idx}.png')
                idx += 1
