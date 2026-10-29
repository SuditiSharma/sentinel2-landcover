"""Stress tests on the trained CNNs: missing bands, noise, and confidence on mistakes."""
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from cnn import load_model, predict
from data import BANDS, ROOT, RGB, load, split

OUT = ROOT / "results"

# band groups to knock out (set to the training mean, which is 0 after normalising)
GROUPS = {
    "none": [],
    "coastal+vapour+cirrus (B01,B09,B10)": ["B01", "B09", "B10"],
    "visible (B02,B03,B04)": ["B02", "B03", "B04"],
    "red edge (B05,B06,B07)": ["B05", "B06", "B07"],
    "NIR (B08,B8A)": ["B08", "B8A"],
    "SWIR (B11,B12)": ["B11", "B12"],
}


def drop(channels):
    def f(x):
        x = x.clone()
        x[:, channels] = 0.0
        return x
    return f


def add_noise(sigma, seed=0):
    g = torch.Generator().manual_seed(seed)

    def f(x):
        return x + sigma * torch.randn(x.shape, generator=g)
    return f


def main():
    X, y, classes = load()
    _, te = split(y)
    rows = []

    # 1. band knock-out on the 13-band model
    model, bands, mean, std = load_model("all13")
    for gname, names in GROUPS.items():
        ch = [BANDS.index(n) for n in names]
        _, yt, p = predict(model, X, te, y, bands, mean, std, transform=drop(ch))
        per_class = {c: (p.argmax(1)[yt == i] == i).mean() for i, c in enumerate(classes)}
        rows.append({"test": "drop bands", "setting": gname, "model": "all13",
                     "acc": (p.argmax(1) == yt).mean(), **per_class})
        print("drop", gname, round((p.argmax(1) == yt).mean(), 4), flush=True)

    # 2. gaussian noise (in standard-deviation units of each band) for both models
    for mname in ["rgb", "all13"]:
        model, bands, mean, std = load_model(mname)
        for sigma in [0.0, 0.1, 0.25, 0.5, 1.0]:
            _, yt, p = predict(model, X, te, y, bands, mean, std, transform=add_noise(sigma))
            per_class = {c: (p.argmax(1)[yt == i] == i).mean() for i, c in enumerate(classes)}
            rows.append({"test": "noise", "setting": f"sigma={sigma}", "model": mname,
                         "acc": (p.argmax(1) == yt).mean(), **per_class})
            print("noise", mname, sigma, round((p.argmax(1) == yt).mean(), 4), flush=True)

    df = pd.DataFrame(rows)
    df.round(3).to_csv(OUT / "failure_tests.csv", index=False)

    # 3. how confident is the clean 13-band model when it is wrong?
    model, bands, mean, std = load_model("all13")
    idx, yt, p = predict(model, X, te, y, bands, mean, std)
    pred, conf = p.argmax(1), p.max(1)
    wrong = pred != yt
    pairs = Counter((classes[a], classes[b]) for a, b in zip(yt[wrong], pred[wrong]))
    conf_info = {
        "n_test": int(len(yt)),
        "n_wrong": int(wrong.sum()),
        "median_conf_correct": float(np.median(conf[~wrong])),
        "median_conf_wrong": float(np.median(conf[wrong])),
        "share_wrong_with_conf_over_0.9": float((conf[wrong] > 0.9).mean()),
        "top_confusions": [[t, pr, n] for (t, pr), n in pairs.most_common(8)],
    }
    (OUT / "confidence.json").write_text(json.dumps(conf_info, indent=2))
    print(json.dumps(conf_info, indent=2))

    fig, ax = plt.subplots(figsize=(6, 3.5))
    bins = np.linspace(0.1, 1, 30)
    ax.hist(conf[~wrong], bins=bins, alpha=0.6, density=True, label="correct")
    ax.hist(conf[wrong], bins=bins, alpha=0.6, density=True, label="wrong")
    ax.set_xlabel("top softmax probability")
    ax.set_ylabel("density")
    ax.set_title("13-band CNN confidence on the test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "confidence_hist.png", dpi=130)
    plt.close(fig)

    # the most confident mistakes, shown as true colour
    worst = np.argsort(-(conf * wrong))[:12]
    fig, axes = plt.subplots(2, 6, figsize=(12, 4.6))
    for ax, k in zip(axes.flat, worst):
        rgb = np.asarray(X[idx[k]][RGB], dtype=np.float32).transpose(1, 2, 0)
        rgb = np.clip(rgb / 3000, 0, 1)
        ax.imshow(rgb)
        ax.set_title(f"true {classes[yt[k]]}\npred {classes[pred[k]]} {conf[k]:.2f}", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "confident_mistakes.png", dpi=130)
    plt.close(fig)

    # noise curve
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for mname in ["rgb", "all13"]:
        d = df[(df.test == "noise") & (df.model == mname)]
        ax.plot([float(s.split("=")[1]) for s in d.setting], d.acc, marker="o", label=mname)
    ax.set_xlabel("noise sigma (band std units)")
    ax.set_ylabel("test accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "noise_curve.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
