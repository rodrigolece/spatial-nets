import os
from pathlib import Path
import pickle

#  import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from spatial_nets import LocationsDataClass
from spatial_nets import utils
from spatial_nets import draw

from config import SAVEFIG, FORMAT


# Change the default colors
colors = ["aqua", "plum", "gainsboro"]
custom_cycler = cycler(color=(draw.named_colors[c] for c in colors))
plt.rcParams.update({"axes.prop_cycle": custom_cycler})
plt.rcParams.update({"font.size": 16})


data_dir = Path(os.environ["workdirc"]) / "data"
output_dir = Path("output_figures")

dmat = utils.load_dmat(data_dir / "UK_geodesic_dmat.npz")
data = utils.load_flows(data_dir / "UK_commute2011.npz")
locs = LocationsDataClass(data, coords=dmat)

print("\nLoading DC gravity data\n")
descriptor = "geo_grav_doubly"
#  prediction = np.load(output_dir / f"model_{descriptor}.npy")
with open(output_dir / f"pvalues_{descriptor}.pkl", "rb") as f:
    pvals_grav = pickle.load(f)
    pvals_grav.set_significance(0.01)


print("\nLoading PC Radiation data\n")
descriptor = "geo_rad_prod"
#  prediction = np.load(output_dir / f"model_{descriptor}.npy")
with open(output_dir / f"pvalues_{descriptor}.pkl", "rb") as f:
    pvals_rad = pickle.load(f)
    pvals_rad.set_significance(0.01)


fig, axes = plt.subplots(figsize=(5, 5), nrows=2, sharex=True)
ax = axes[0]
draw.signed_distance_histogram(locs, pvals_grav, ax, verbose=True)
ax.set_yticks([0, 5_000, 10_000, 15_000])
ax.ticklabel_format(axis="y", style="sci", scilimits=(3, 3))
ax.set_ylabel("Edge counts")
ax.set_title("Gravity DC")
ax.legend(loc="upper left", fontsize=12)

ax = axes[1]
draw.signed_distance_histogram(locs, pvals_rad, ax, verbose=True)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.set_xlabel("Distance")
ax.set_ylabel("Edge counts")
ax.set_title("Radiation PC")

if SAVEFIG:
    filename = f"histograms_distance.{FORMAT}"
    print("Writing: ", filename)
    fig.savefig(output_dir / filename, bbox_inches="tight")
