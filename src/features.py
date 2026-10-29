"""Hand-made features for each patch: band means/stds plus three indices."""
import numpy as np

from data import BANDS

B = {name: i for i, name in enumerate(BANDS)}


def norm_diff(a, b):
    return (a - b) / (a + b + 1e-6)


def patch_features(X, chunk=2000):
    """X is (N,13,64,64). Returns (N, 32) features and their names."""
    out = []
    for s in range(0, len(X), chunk):
        x = np.asarray(X[s:s + chunk], dtype=np.float32)
        ndvi = norm_diff(x[:, B["B08"]], x[:, B["B04"]])   # vegetation
        ndwi = norm_diff(x[:, B["B03"]], x[:, B["B08"]])   # open water
        ndbi = norm_diff(x[:, B["B11"]], x[:, B["B08"]])   # built-up / bare
        parts = [x.mean(axis=(2, 3)), x.std(axis=(2, 3))]
        for idx in (ndvi, ndwi, ndbi):
            parts.append(idx.mean(axis=(1, 2))[:, None])
            parts.append(idx.std(axis=(1, 2))[:, None])
        out.append(np.concatenate(parts, axis=1))
    names = ([f"{b}_mean" for b in BANDS] + [f"{b}_std" for b in BANDS]
             + ["ndvi_mean", "ndvi_std", "ndwi_mean", "ndwi_std", "ndbi_mean", "ndbi_std"])
    return np.concatenate(out), names
