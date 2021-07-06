#  import os
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as cs

#  from spatial_nets import draw

from config import SAVEFIG, FORMAT


set3 = list(plt.get_cmap("Set3").colors)
set3.pop(1)  # the same Peixoto does in graph-tool
cm = colors.LinearSegmentedColormap.from_list("gt-Set3", set3)


raw_data_dir = Path("raw")
data_dir = Path("output_data")
output_dir = Path("output_figures")

coords: np.ndarray = np.load(raw_data_dir / "UK_centroids.npz")["longlat"]

b_files = {
    "b_geo_grav-doubly_li_10.npy": "li",
    "b_geo_grav-doubly_lc_10.npy": "lc",
    "b_geo_grav-doubly_wc_10.npy": "wc",
}
# We shift the vector b to get better colors for the regions of interest
shift = [3, 3, 4]  # 3 for li, lc and 4 for wc

for k, (f, extension) in enumerate(b_files.items()):
    b: np.ndarray = np.load(data_dir / f)
    B = b.max()
    c = (b + shift[k]) % (B + 1)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection=cs.Mercator()))

    ax.coastlines(resolution="auto", color="0.7")
    ax.set_extent((-10, 2.5, 49.5, 61))
    # ax.gridlines(draw_labels=True)

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=40,
        c=c,
        alpha=0.8,
        #  edgecolos="w",
        cmap=cm,
        transform=cs.PlateCarree(),
    )

    if SAVEFIG:
        filename = output_dir / f"map_uk-commuting_{extension}.{FORMAT}"
        print("Saving map: ", filename)
        fig.savefig(filename, bbox_inches="tight")


# For reference, from old map
#  Lambert Conformal basemap.
#  m = Basemap(width=int(1e6), height=int(1.3e6), projection='lcc',
#              resolution=res, lat_1=50., lat_2=60, lat_0=55, lon_0=-3.85)
