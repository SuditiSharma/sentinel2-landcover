"""Load EuroSAT (13-band) into one numpy array and make a fixed train/test split."""
from pathlib import Path

import numpy as np
import tifffile
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"

# Band order inside the EuroSAT_MS tif files (B8A comes last)
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
         "B08", "B09", "B10", "B11", "B12", "B8A"]
RGB = [BANDS.index("B04"), BANDS.index("B03"), BANDS.index("B02")]


def find_class_dirs():
    # the zip unpacks to EuroSAT_MS/<ClassName>/*.tif
    base = next(p for p in RAW.rglob("*") if p.is_dir() and (p / "Forest").is_dir())
    return sorted(d for d in base.iterdir() if d.is_dir())


def build_cache():
    """Read every tif once and save X (N,13,64,64 uint16) and y to disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    class_dirs = find_class_dirs()
    classes = [d.name for d in class_dirs]
    files, labels = [], []
    for i, d in enumerate(class_dirs):
        for f in sorted(d.glob("*.tif")):
            files.append(f)
            labels.append(i)

    X = np.lib.format.open_memmap(CACHE / "X.npy", mode="w+", dtype=np.uint16,
                                  shape=(len(files), 13, 64, 64))
    for n, f in enumerate(files):
        img = tifffile.imread(f)            # (64, 64, 13)
        X[n] = np.moveaxis(img, -1, 0)
    X.flush()
    np.save(CACHE / "y.npy", np.array(labels, dtype=np.int64))
    (CACHE / "classes.txt").write_text("\n".join(classes))
    print(f"cached {len(files)} images, {len(classes)} classes")


def load():
    if not (CACHE / "X.npy").exists():
        build_cache()
    X = np.load(CACHE / "X.npy", mmap_mode="r")
    y = np.load(CACHE / "y.npy")
    classes = (CACHE / "classes.txt").read_text().split("\n")
    return X, y, classes


def split(y, seed=0):
    """Same 80/20 stratified split for every model."""
    idx = np.arange(len(y))
    return train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
