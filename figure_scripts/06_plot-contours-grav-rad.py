import os
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

#  from matplotlib.colors import Normalize
#  from mpl_toolkits.axes_grid1 import make_axes_locatable
#  from matplotlib import cm

from spatial_nets import draw

from config import SAVEFIG, FORMAT

plt.rcParams.update({"font.size": 20})
cmaps = draw.setup_default_colormaps()
#  plt.set_cmap("Greys_r")


data_dir = Path("output_data")
output_dir = Path("output_figures")


var = "nmi"  # nmi, Bs, nmif
#  normalize the color scale for the NMI
#  norm = Normalize(0.0, 1.0)
#  Below is used instead of norm
kwargs = {}
levels = None
if var != "Bs":
    nb_levels = 10
    levels = np.linspace(0, 1, nb_levels + 1)
    kwargs.update(dict(levels=levels, vmin=0, vmax=1))

clabel = var.upper()
if var != "nmi":
    clabel = clabel[:-1]
labels = [r"$\rho$", r"$\lambda$", clabel]

files = {
    # Gravity
    "expert-rho-lamb_gamma2_10_10_grav.npz": "melrose_hls",
    "expert-rho-lamb_gamma1_10_10_grav.npz": "melrose_hls",
    # Rad, for the time being small matrices
    "expert-rho-lamb_gamma2_02_02_rad.npz": "shakespeare_hls",
    "expert-rho-lamb_gamma1_02_02_rad.npz": "shakespeare_hls",
    #  "expert-rho-lamb_gamma2_10_10_rad.npz": "shakespeare_hls",
    #  "expert-rho-lamb_gamma1_10_10_rad.npz": "shakespeare_hls",
}
cbars = [True, False, True, False]

for k, (f, cname) in enumerate(files.items()):
    base, ext = os.path.splitext(f)
    plt.set_cmap(cmaps[cname].reversed())

    data = np.load(data_dir / f)

    # We remove values that are slightly larger than 1 (for NMI artifacts)
    Z = data[var]
    if levels is not None and (diff := levels[-1] - np.max(Z)) < 0:
        if np.abs(diff) < 1e-6:
            print("Removing small artifact")
            Z = np.maximum(Z + diff, 0)

    fig, ax = plt.subplots(figsize=(5, 4))
    draw.contourf(
        data["rho"],
        data["lamb"],
        Z,
        ax,
        fig,
        labels=labels,
        colorbar=cbars[k],
        **kwargs,
    )
    ax.set_xscale("log")

    # Corlorbar normalized btwn [0, 1]
    #  Better not to use this because you loose the levels created by contourf
    #  divider = make_axes_locatable(ax)
    #  cax = divider.append_axes("right", size="5%", pad=0.3)
    #  fig.colorbar(cm.ScalarMappable(norm=norm), cax=cax)
    #  cax.set_ylabel(labels[2], labelpad=10)

    #  ax.set_title(base)

    if SAVEFIG:
        descriptor = base.split("_")[1:]
        model = descriptor.pop(-1)
        descriptor = "_".join(descriptor)
        fix = "-fix" if var == "nmif" else ""
        filename = f"contourf_{model}{fix}_{descriptor}.{FORMAT}"

        print("Saving to: ", filename)
        fig.savefig(
            output_dir / filename,
            bbox_inches="tight",
        )
