import os
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

coords = np.load(raw_data_dir / "UK_centroids.npz")["longlat"]

#  b = np.load(data_dir / "b_geo_grav-doubly_li_10.npy")
#  b = np.load(data_dir / "b_geo_grav-doubly_lc_10.npy")
b = np.load(data_dir / "b_geo_grav-doubly_wc_10.npy")

# We shift the vector b to get better colors for the regions of interest
B = b.max()
c = (b + 4) % (B + 1)  # 3 for li, lc and 4 for wc

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

#  filename = "map_uk-commuting_li.svg"
#  filename = "map_uk-commuting_lc.svg"
filename = "map_uk-commuting_wc.svg"
fig.savefig(output_dir / filename, bbox_inches="tight")


# For reference, from old map
#  Lambert Conformal basemap.
#  m = Basemap(width=int(1e6), height=int(1.3e6), projection='lcc',
#              resolution=res, lat_1=50., lat_2=60, lat_0=55, lon_0=-3.85)
