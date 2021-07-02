import os
from pathlib import Path

import numpy as np
from scipy.io import loadmat

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


var = "nmi"  # nmi_std, Bs, Bs_std
#  normalize the color scale for the NMI
#  norm = Normalize(0.0, 1.0)
#  Below is used instead of norm
kwargs = {}
if var != "Bs":
    nb_levels = 10
    levels = np.linspace(0, 1, nb_levels + 1)
    kwargs.update(dict(levels=levels, vmin=0, vmax=1))

labels = [r"$\rho$", r"$\lambda$", var.upper()]

files = {
    "mod-expert-rho-lamb_gamma2_binsize2_1_10.mat": "rajah_hls",
    "mod-expert-rho-lamb_gamma1_binsize2_1_10.mat": "rajah_hls",
}
cbars = [True, False]

#  for bins in [1, 2, 5]:
for k, (f, cname) in enumerate(files.items()):
    base, ext = os.path.splitext(f)
    plt.set_cmap(cmaps[cname].reversed())

    data = loadmat(data_dir / f)

    fig, ax = plt.subplots(figsize=(5, 4))
    draw.contourf(
        data["rho"],
        data["lamb"],
        data[var],
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
        descriptor = "_".join(base.split("_")[1:])
        filename = f"contourf_mod_{descriptor}.{FORMAT}"
        print("Saving to: ", filename)
        fig.savefig(output_dir / filename, bbox_inches="tight")
