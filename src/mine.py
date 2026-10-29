"""Pixel random forest on the Bingham Canyon scene using the hand-drawn boxes."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from data import ROOT
from make_labels import BOXES, CLASSES, OUT as LABELS, SCENE
from plots import plot_confusion

OUT = ROOT / "results" / "mine"
COLOURS = ["#6b3e26", "#c9a66b", "#2e7d32", "#1e88e5", "#e53935", "#e8d9a8"]
BAND_ORDER = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]


def pixel_features(a):
    """a is (10, H, W) reflectance x 10000. Returns (H*W, 13)."""
    a = a.astype(np.float32)
    b = {n: a[i] for i, n in enumerate(BAND_ORDER)}
    nd = lambda x, y: (x - y) / (x + y + 1e-6)
    extra = np.stack([nd(b["B08"], b["B04"]), nd(b["B03"], b["B08"]), nd(b["B11"], b["B08"])])
    return np.concatenate([a / 10000, extra]).reshape(13, -1).T


def stretch(rgb):
    lo, hi = np.percentile(rgb, [2, 99.5], axis=(1, 2))
    return np.clip((rgb - lo[:, None, None]) / (hi - lo)[:, None, None], 0, 1).transpose(1, 2, 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with rasterio.open(SCENE) as src:
        a = src.read()
        transform, shape = src.transform, (src.height, src.width)
    F = pixel_features(a)

    gj = json.loads(LABELS.read_text())
    masks = {}
    for split in ["train", "test"]:
        shapes = [(f["geometry"], CLASSES.index(f["properties"]["class"]) + 1)
                  for f in gj["features"] if f["properties"]["split"] == split]
        masks[split] = rasterize(shapes, out_shape=shape, transform=transform, fill=0).ravel()

    rng = np.random.default_rng(0)
    tr = np.flatnonzero(masks["train"])
    # cap each class at 4000 training pixels so the big boxes don't swamp water
    keep = np.concatenate([rng.permutation(tr[masks["train"][tr] == k])[:4000]
                           for k in range(1, len(CLASSES) + 1)])
    te = np.flatnonzero(masks["test"])
    y_tr, y_te = masks["train"][keep] - 1, masks["test"][te] - 1
    counts = {c: [int((y_tr == i).sum()), int((y_te == i).sum())] for i, c in enumerate(CLASSES)}
    print("pixels per class [train, test]:", counts)

    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0,
                                class_weight="balanced", min_samples_leaf=2)
    rf.fit(F[keep], y_tr)
    pred_te = rf.predict(F[te])
    acc = accuracy_score(y_te, pred_te)
    print("held-out box accuracy", round(acc, 4))
    print(classification_report(y_te, pred_te, target_names=CLASSES, digits=3, zero_division=0))
    pd.DataFrame(classification_report(y_te, pred_te, target_names=CLASSES, output_dict=True,
                                       zero_division=0)).T.round(3).to_csv(OUT / "mine_report.csv")
    plot_confusion(y_te, pred_te, CLASSES, f"mine RF, held-out boxes (acc {acc:.3f})",
                   OUT / "cm_mine.png")

    # predict the whole scene in chunks
    pred = np.concatenate([rf.predict(F[s:s + 500_000]) for s in range(0, len(F), 500_000)])
    pred_map = pred.reshape(shape)
    np.save(OUT / "pred_map.npy", pred_map.astype(np.uint8))
    area = {c: round(float((pred_map == i).mean()), 4) for i, c in enumerate(CLASSES)}
    imp = dict(zip(BAND_ORDER + ["ndvi", "ndwi", "ndbi"], rf.feature_importances_.round(3).tolist()))
    (OUT / "mine_summary.json").write_text(json.dumps(
        {"heldout_acc": round(acc, 4), "pixels_train_test": counts, "map_share": area,
         "feature_importance": imp}, indent=2))
    print("share of scene per class:", area)

    rgb = stretch(a[[2, 1, 0]].astype(np.float32))
    fig, axes = plt.subplots(1, 2, figsize=(11, 8.6))
    axes[0].imshow(rgb)
    for cls, boxes in BOXES.items():
        for split, c0, r0, c1, r1 in boxes:
            axes[0].add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, lw=1.2,
                                        ls="-" if split == "train" else "--",
                                        ec=COLOURS[CLASSES.index(cls)]))
    axes[0].set_title("Sentinel-2 true colour, 6 Sep 2024\nboxes: solid = train, dashed = test", fontsize=9)
    axes[1].imshow(pred_map, cmap=ListedColormap(COLOURS), vmin=-0.5, vmax=len(CLASSES) - 0.5,
                   interpolation="nearest")
    axes[1].set_title("random forest prediction", fontsize=9)
    axes[1].legend(handles=[Patch(color=c, label=n) for c, n in zip(COLOURS, CLASSES)],
                   loc="lower right", fontsize=7)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "mine_map.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
