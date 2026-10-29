import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion(y_true, y_pred, classes, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return cm
