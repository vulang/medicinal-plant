import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools

def resolve_device(cfg_device: str = "auto"):
    if cfg_device == "cpu":
        return torch.device("cpu")
    if cfg_device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if cfg_device == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    # annotate
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

def save_classification_report(y_true, y_pred, classes, save_path):
    labels = list(range(len(classes)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=classes,
        digits=4,
        zero_division=0,
    )
    with open(save_path, "w") as f:
        f.write(report)
