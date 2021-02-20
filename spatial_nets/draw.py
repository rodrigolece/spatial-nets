import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import colorsys
from mpl_toolkits.axes_grid1 import make_axes_locatable

from spatial_nets import utils


# default_cm = gt.default_cm  # 'Set3'
# The colors below come from plt.get_cmap('Set3').colors

set3_colors = list(plt.get_cmap('Set3').colors)
set3_colors.pop(1)  # the same as Peixoto does in graph-tool

default_cm = colors.LinearSegmentedColormap.from_list(
    'graphtool-Set3',
    set3_colors
)

default_names = [
    'acqua', 'melrose', 'salmon', 'shakespeare', 'rajah', 'sulu',
     'classic_rose', 'gainsboro', 'plum', 'snowy_mint', 'witch_haze'
]



def display_cmap(cmap, N=20, ax=None):
    """Display a colormap using N bins."""
    x = np.linspace(0, 1, N)
    x = np.vstack((x, x))

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(x, cmap=cmap)
    ax.axis('off')
    ax.set_title(cmap.name)

    return None


def hsv_cmap_from_color(color, min_value, max_value, name):
    """Create a colormap from a given seed color in HSV color space."""
    h, s, v = colors.rgb_to_hsv(color)
    dark = colors.hsv_to_rgb((h, s, min_value))
    light = colors.hsv_to_rgb((h, s, max_value))
    cmap = colors.LinearSegmentedColormap.from_list(name+'_hsv', [dark, light])

    return cmap


def hls_cmap_from_color(color, min_value, max_value, name):
    """Create a colormap from a given seed color in HLS color space."""
    h, l, s = colorsys.rgb_to_hls(*color)
    dark = colorsys.hls_to_rgb(h, min_value, s)
    light = colorsys.hls_to_rgb(h, max_value, s)
    cmap = colors.LinearSegmentedColormap.from_list(name+'_hls', [dark, light])

    return cmap


def setup_default_colormaps(register=True):
    """Create the HSV and HLS colormaps based on the default Set3 colors."""
    out = {}

    for k, c in enumerate(set3_colors):
        hsv = hsv_cmap_from_color(c, 0.5, 1.0, default_names[k])
        out[hsv.name] = hsv

        hls = hls_cmap_from_color(c, 0.5, 1.0, default_names[k])
        out[hls.name] = hls

        if register:
            cm.register_cmap(name=hsv.name, cmap=hsv)
            cm.register_cmap(name=hls.name, cmap=hls)

            # also register reversed colormaps
            cm.register_cmap(name=hsv.name + '_r', cmap=hsv.reversed())
            cm.register_cmap(name=hls.name + '_r', cmap=hls.reversed())


    return out


def gt_color_legend(state, legendsize=(6, 0.35), cmap=default_cm):
    """Axis with discrete colors corresponding to GraphTool.BlockState object."""
    nb_colors = state.get_B()
    gradient = np.linspace(0, 1, nb_colors)
    gradient = np.vstack((gradient, gradient))

    fig, ax = plt.subplots(figsize=legendsize, squeeze=True)
    ax.imshow(gradient, aspect='auto', cmap=cmap)
    ax.get_xaxis().set_ticks(range(nb_colors))
    ax.get_yaxis().set_visible(False)
    # ax.set_axis_off()

    return None


def signed_scatterplot(locs,
                       T_model,
                       idx_plus,
                       idx_minus,
                       ax,
                       colors=['C0', 'C1', '0.75'],
                       alpha=0.2,
                       fs=18,
                       rounded=False,
                       threshold=None):

    i, j = locs.data.nonzero()
    observed = np.asarray(locs.data[i, j]).flatten()
    predicted = T_model[i, j]

    plus_observed = observed[idx_plus]
    plus_predicted = predicted[idx_plus]

    minus_observed = observed[idx_minus]
    minus_predicted = predicted[idx_minus]

    idx_zero = ~np.bitwise_or(idx_plus, idx_minus)

    zero_observed = observed[idx_zero]
    zero_predicted = predicted[idx_zero]

    if rounded:
        plus_predicted = plus_predicted.round()
        minus_predicted = minus_predicted.round()
        zero_predicted = zero_predicted.round()

    if threshold:
        plus_predicted[plus_predicted < threshold] = 0.0
        minus_predicted[minus_predicted < threshold] = 0.0
        zero_predicted[zero_predicted < threshold] = 0.0

    ax.plot(minus_observed, minus_predicted, '.',
            color=colors[0], alpha=alpha, label='negative')
    ax.plot(plus_observed, plus_predicted, '.',
            color=colors[1], alpha=alpha, label='positive')
    ax.plot(zero_observed, zero_predicted, '.',
            color=colors[2], alpha=alpha, label='not significant')

    x, X, y, Y  = ax.axis()
    mM = max(x,y), min(X,Y)
    ax.plot(mM, mM, ls='--', c='.3', alpha=0.5, label='y=x (diagonal)')
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Observed flow', fontsize=fs)
    ax.set_ylabel('Predicted flow', fontsize=fs)

    return None


def contourf(x, y, z, ax, fig, labels=None, colorbar=True, norm=None):
    im = ax.contourf(x, y, z, norm=norm)

    labels = utils._get_iterable(labels)
    if len(labels) > 1:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.3)
        fig.colorbar(im, cax=cax)
        if len(labels) == 3:
            cax.set_ylabel(labels[2], labelpad=10)

    return fig, ax
