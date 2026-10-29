"""Write the hand-drawn label boxes for the mine scene to a GeoJSON file.

I drew these by looking at a contrast-stretched true-colour image with a pixel
grid on top and noting the (col, row) corners. They are plain rectangles, not
careful outlines. The GeoJSON is in the scene CRS (UTM 12N) so it can be opened
and fixed in QGIS.

Each class has a "train" box and a separate "test" box somewhere else in the
scene, so the test score is not just pixels next to the training pixels.
"""
import json

import rasterio

from data import ROOT

SCENE = ROOT / "data" / "mine" / "bingham_20240906.tif"
OUT = ROOT / "data" / "mine" / "labels.geojson"

# class: [(split, col0, row0, col1, row1), ...]
BOXES = {
    "pit": [("train", 230, 2250, 330, 2350), ("test", 300, 2120, 400, 2200)],
    "waste_rock": [("train", 440, 1650, 540, 1760), ("test", 620, 1950, 680, 2150)],
    "vegetation": [("train", 50, 150, 200, 300), ("test", 250, 350, 380, 480)],
    "water": [("train", 930, 1472, 948, 1492), ("test", 967, 1458, 978, 1472)],
    "built_up": [("train", 1480, 200, 1640, 350), ("test", 1720, 1600, 1840, 1700)],
    "rangeland": [("train", 800, 1750, 950, 1900), ("test", 700, 350, 850, 500)],
}
CLASSES = list(BOXES)


def main():
    with rasterio.open(SCENE) as src:
        t, crs = src.transform, src.crs.to_string()
    feats = []
    for cls, boxes in BOXES.items():
        for split, c0, r0, c1, r1 in boxes:
            corners = [t * (c0, r0), t * (c1, r0), t * (c1, r1), t * (c0, r1), t * (c0, r0)]
            feats.append({"type": "Feature",
                          "properties": {"class": cls, "split": split},
                          "geometry": {"type": "Polygon", "coordinates": [corners]}})
    gj = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs}},
          "features": feats}
    OUT.write_text(json.dumps(gj, indent=1))
    print("wrote", len(feats), "boxes to", OUT)


if __name__ == "__main__":
    main()
