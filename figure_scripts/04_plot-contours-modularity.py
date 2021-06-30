import os
from pathlib import Path

#  import numpy as np
from scipy.io import loadmat

from spatial_nets import draw

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm

from config import SAVEFIG, FORMAT

plt.rcParams.update({"font.size": 14})
plt.set_cmap("Greys_r")


data_dir = Path(os.environ["workdirc"]) / "validation" / "output_modularity"
output_dir = Path("output")


var = "nmi"  # Bs or nmi_std, Bs_std
labels = [r"$\rho$", r"$\lambda$", var.upper()]
norm = Normalize(0.0, 1.0)

for bins in [1, 2, 5]:
    data = loadmat(data_dir / f"binsize{bins:02}_1_10.mat")

    fig, ax = plt.subplots(figsize=(5, 4))
    draw.contourf(
        data["rho"],
        data["lamb"],
        data[var],
        ax,
        fig,
        labels=labels,
        colorbar=False,
        norm=norm,
    )
    ax.set_xscale("log")

    # Corlorbar normalized btwn [0, 1]
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    fig.colorbar(cm.ScalarMappable(norm=norm), cax=cax)
    cax.set_ylabel(labels[2], labelpad=10)
    #  ax.set_title()

    if SAVEFIG:
        filename = f"contourf_modularity_bins{bins:02}.{FORMAT}"
        print("Saving to: ", filename)
        fig.savefig(
            output_dir / filename,
            bbox_inches="tight",
        )
