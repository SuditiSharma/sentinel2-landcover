"""Download a clip of one cloud-free Sentinel-2 L2A scene over Bingham Canyon mine, Utah.

Source: Microsoft Planetary Computer, collection sentinel-2-l2a.
The 20 m bands are resampled to 10 m (bilinear) so every band is on the same grid.
"""
import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from data import ROOT

ITEM_ID = "S2B_MSIL2A_20240906T181919_R127_T12TVK_20240907T002550"
BBOX = (-112.22, 40.48, -112.00, 40.74)   # lon/lat: pit, waste dumps, tailings, towns
BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
OUT = ROOT / "data" / "mine" / "bingham_20240906.tif"


def main():
    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)
    item = next(cat.search(collections=["sentinel-2-l2a"], ids=[ITEM_ID]).items())

    with rasterio.open(item.assets["B04"].href) as ref:
        bounds = transform_bounds("EPSG:4326", ref.crs, *BBOX)
        win = from_bounds(*bounds, ref.transform).round_offsets().round_lengths()
        transform = ref.window_transform(win)
        profile = dict(driver="GTiff", height=int(win.height), width=int(win.width), count=len(BANDS),
                       dtype="uint16", crs=ref.crs, transform=transform, compress="deflate")

    stack = []
    for b in BANDS:
        with rasterio.open(item.assets[b].href) as src:
            w = from_bounds(*bounds, src.transform)
            stack.append(src.read(1, window=w, out_shape=(profile["height"], profile["width"]),
                                  resampling=Resampling.bilinear))
        print("read", b, flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(OUT, "w", **profile) as dst:
        dst.write(np.stack(stack))
        dst.descriptions = tuple(BANDS)
    print("saved", OUT, profile["width"], "x", profile["height"])


if __name__ == "__main__":
    main()
