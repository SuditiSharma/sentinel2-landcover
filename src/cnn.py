"""Small CNN trained on RGB only and on all 13 bands, same split and settings."""
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

from data import ROOT, RGB, load, split
from plots import plot_confusion

OUT = ROOT / "results"
MODELS = ROOT / "models"
torch.set_num_threads(4)


class SmallCNN(nn.Module):
    def __init__(self, in_ch, n_classes=10):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.MaxPool2d(2))
        self.features = nn.Sequential(block(in_ch, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def band_stats(X, idx, bands):
    sample = np.asarray(X[np.sort(np.random.default_rng(0).choice(idx, 3000, replace=False))][:, bands],
                        dtype=np.float32)
    return sample.mean(axis=(0, 2, 3)), sample.std(axis=(0, 2, 3))


def get_batch(X, idx, bands, mean, std):
    x = np.asarray(X[np.sort(idx)][:, bands], dtype=np.float32)
    x = (x - mean[None, :, None, None]) / std[None, :, None, None]
    return torch.from_numpy(x)


def predict(model, X, idx, y, bands, mean, std, bs=256, transform=None):
    """Returns sorted indices, true labels and softmax probabilities."""
    idx = np.sort(idx)
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, len(idx), bs):
            xb = get_batch(X, idx[s:s + bs], bands, mean, std)
            if transform is not None:
                xb = transform(xb)
            probs.append(torch.softmax(model(xb), 1).numpy())
    return idx, y[idx], np.concatenate(probs)


def train(name, bands, epochs=12, bs=64):
    X, y, classes = load()
    tr, te = split(y)
    mean, std = band_stats(X, tr, bands)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = SmallCNN(len(bands))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss()

    log = []
    for ep in range(epochs):
        model.train()
        t0, total = time.time(), 0.0
        order = rng.permutation(tr)
        for s in range(0, len(order), bs):
            bidx = np.sort(order[s:s + bs])
            xb = get_batch(X, bidx, bands, mean, std)
            yb = torch.from_numpy(y[bidx])
            if rng.random() < 0.5:                      # random flips
                xb = xb.flip(3)
            if rng.random() < 0.5:
                xb = xb.flip(2)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(bidx)
        sched.step()
        _, yt, p = predict(model, X, te, y, bands, mean, std)
        acc = accuracy_score(yt, p.argmax(1))
        log.append({"epoch": ep + 1, "train_loss": total / len(tr), "test_acc": acc})
        print(f"{name} epoch {ep + 1}: loss {total / len(tr):.3f}  test acc {acc:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    MODELS.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "bands": bands, "mean": mean, "std": std},
               MODELS / f"cnn_{name}.pt")
    _, yt, p = predict(model, X, te, y, bands, mean, std)
    pred = p.argmax(1)
    pd.DataFrame(log).to_csv(OUT / f"cnn_{name}_log.csv", index=False)
    pd.DataFrame(classification_report(yt, pred, target_names=classes, output_dict=True)).T \
        .round(3).to_csv(OUT / f"cnn_{name}_report.csv")
    plot_confusion(yt, pred, classes, f"CNN {name} (acc {accuracy_score(yt, pred):.3f})",
                   OUT / f"cm_cnn_{name}.png")
    print(classification_report(yt, pred, target_names=classes, digits=3))
    return accuracy_score(yt, pred)


def load_model(name):
    ck = torch.load(MODELS / f"cnn_{name}.pt", weights_only=False)
    model = SmallCNN(len(ck["bands"]))
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck["bands"], ck["mean"], ck["std"]


if __name__ == "__main__":
    which = sys.argv[1:] or ["rgb", "all13"]
    results = {}
    for name in which:
        bands = RGB if name == "rgb" else list(range(13))
        results[name] = round(train(name, bands), 4)
    path = OUT / "cnn_summary.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(results)
    path.write_text(json.dumps(old, indent=2))
