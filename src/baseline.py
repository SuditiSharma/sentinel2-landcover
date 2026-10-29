"""Random forest and logistic regression on simple patch features."""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import ROOT, load, split
from features import patch_features
from plots import plot_confusion

OUT = ROOT / "results"


def main():
    OUT.mkdir(exist_ok=True)
    X, y, classes = load()
    tr, te = split(y)

    feat_path = ROOT / "data" / "cache" / "features.npy"
    if feat_path.exists():
        F = np.load(feat_path)
    else:
        F, _ = patch_features(X)
        np.save(feat_path, F)

    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "random_forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0),
    }
    summary = {}
    for name, model in models.items():
        model.fit(F[tr], y[tr])
        pred = model.predict(F[te])
        acc = accuracy_score(y[te], pred)
        report = classification_report(y[te], pred, target_names=classes, output_dict=True)
        pd.DataFrame(report).T.round(3).to_csv(OUT / f"{name}_report.csv")
        plot_confusion(y[te], pred, classes, f"{name} (acc {acc:.3f})", OUT / f"cm_{name}.png")
        summary[name] = round(acc, 4)
        print(name, round(acc, 4))
        print(classification_report(y[te], pred, target_names=classes, digits=3))

    (OUT / "baseline_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
