import os
from pathlib import Path
import pickle

import numpy as np
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


data_dir = Path(os.environ["workdirc"]) / "data"
output_dir = Path("output_figures")

dmat = utils.load_dmat(data_dir / "UK_geodesic_dmat.npz")
data = utils.load_flows(data_dir / "UK_commute2011.npz")
locs = LocationsDataClass(data, coords=dmat)

print("\nLoading DC gravity data\n")
descriptor = "geo_grav_doubly"
prediction = np.load(output_dir / f"model_{descriptor}.npy")
with open(output_dir / f"pvalues_{descriptor}.pkl", "rb") as f:
    pvals = pickle.load(f)
    pvals.set_significance(0.01)

# First plot
fig, ax = plt.subplots(figsize=(5, 5))
draw.signed_scatterplot(locs, prediction, pvals, ax, verbose=True)
if SAVEFIG:
    filename = f"scatter_{descriptor}.{FORMAT}"
    print("Writing: ", filename)
    fig.savefig(output_dir / filename, bbox_inches="tight")


print("\nLoading PC Radiation data\n")
descriptor = "geo_rad_prod"
prediction = np.load(output_dir / f"model_{descriptor}.npy")
with open(output_dir / f"pvalues_{descriptor}.pkl", "rb") as f:
    pvals = pickle.load(f)
    pvals.set_significance(0.01)


# Second plot
fig, ax = plt.subplots(figsize=(5, 5))
draw.signed_scatterplot(locs, prediction, pvals, ax, verbose=True)
if SAVEFIG:
    filename = f"scatter_{descriptor}.{FORMAT}"
    print("Writing: ", filename)
    fig.savefig(output_dir / filename, bbox_inches="tight")
