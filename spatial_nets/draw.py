import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from spatial_nets import utils


# default_cm = gt.default_cm  # 'Set3'
# The colors below come from plt.get_cmap('Set3').colors

default_clrs = [(0.5529411764705883, 0.8274509803921568, 0.7803921568627451, 1.0),
                # (1.0, 1.0, 0.7019607843137254, 1.0),
                (0.7450980392156863, 0.7294117647058823, 0.8549019607843137, 1.0),
                (0.984313725490196, 0.5019607843137255, 0.4470588235294118, 1.0),
                (0.5019607843137255, 0.6941176470588235, 0.8274509803921568, 1.0),
                (0.9921568627450981, 0.7058823529411765, 0.3843137254901961, 1.0),
                (0.7019607843137254, 0.8705882352941177, 0.4117647058823529, 1.0),
                (0.9882352941176471, 0.803921568627451, 0.8980392156862745, 1.0),
                (0.8509803921568627, 0.8509803921568627, 0.8509803921568627, 1.0),
                (0.7372549019607844, 0.5019607843137255, 0.7411764705882353, 1.0),
                (0.8, 0.9215686274509803, 0.7725490196078432, 1.0),
                (1.0, 0.9294117647058824, 0.43529411764705883, 1.0)]

default_cm = colors.LinearSegmentedColormap.from_list(
    'graphtool-Set3', default_clrs)



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
