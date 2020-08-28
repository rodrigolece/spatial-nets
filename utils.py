import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import loadmat

from sklearn.metrics import pairwise

import matplotlib.pyplot as plt
from matplotlib import colors


__all__ = ['gt_color_legend',
           'sparsemat_from_flow', 'sparsemat_remove_diag',
           'load_dmat', 'load_flows'
           'benchmark_cerina', 'greatcircle_distance']

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


def sparsemat_from_flow(flow_df, return_ids=False):
    """Convert flow DataFrame to sparse matrix."""
    idx_i, ids = pd.factorize(flow_df.origin, sort=True)
    idx_j, _ = pd.factorize(flow_df.destination, sort=True)
    data = flow_df.flow
    n = len(ids)

    mat = sparse.csr_matrix((data, (idx_i, idx_j)), shape=(n, n))

    return mat, ids if return_ids else mat


def sparsemat_remove_diag(spmat):
    """Create a copy of sparse matrix removing the diagonal entries."""
    mat = spmat.copy()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mat[np.diag_indices_from(mat)] = 0
        mat.eliminate_zeros()

    return mat


# def total_flows(flow_df, remove_selfflow=False):
#     """Calculate total flows by summing over origins and destinations."""
#     if remove_selfflow:
#         idx = (flow_df.origin == flow_df.destination)
#         view = flow_df[~idx]
#     else:
#         view = flow_df
#
#     outflows = view.groupby(by='origin')['flow'].sum().fillna(0)
#     # groupby by default sorts the keys; fillna not needed but safer
#     inflows = view.groupby(by='destination')['flow'].sum().fillna(0)
#
#     return outflows, inflows


# def flow_from_model(mat, ϵ, ids=None, convert_to_int=True):
#     """
#     Put output of a model in a DataFrame flow format.
#
#     ϵ is the threshold above which flows are kept.
#     """
#     n = mat.shape[0]
#
#     if ids is None:
#         ids = np.array(range(n))
#     else:
#         assert len(ids) == n, 'length of ids should match mat shape'
#
#     i, j = np.where(mat > ϵ)
#     data = mat[i, j]
#
#     if convert_to_int:
#         data = np.round(data).astype(int)
#
#     df = pd.DataFrame({'origin': ids[i], 'destination': ids[j], 'flow': data})
#
#     return df


def benchmark_cerina(nb_nodes, edge_density, l, β, ϵ, L=1.0, seed=0, verbose=False):
    """Create a benchmark network of the type proposed by Cerina et al."""
    N = nb_nodes
    nb_edges = nb_nodes * (nb_nodes - 1) * edge_density / 2

    rng = np.random.RandomState(seed)

    # Coordinates
    ds = rng.exponential(scale=l, size=N)
    αs = 2 * np.pi * rng.rand(N)
    shift = L * np.ones(N)
    shift[nb_nodes // 2 + 1:] *= -1

    xs = ds * np.cos(αs) + shift
    ys = ds * np.sin(αs)
    coords = np.vstack((xs, ys)).T

    # Attibute assignment
    idx_plane = xs > 0
    idx_success = rng.rand(N) < 1 - ϵ

    comm_vec = np.zeros((N, 1), dtype=int)  # column vector
    comm_vec[np.bitwise_and(idx_plane, idx_success)] = 1
    comm_vec[np.bitwise_and(idx_plane, ~idx_success)] = -1
    comm_vec[np.bitwise_and(~idx_plane, idx_success)] = -1
    comm_vec[np.bitwise_and(~idx_plane, ~idx_success)] = 1

    # Edge selection
    smat = comm_vec.T * comm_vec
    dmat = pairwise.euclidean_distances(coords)
    pmat = np.triu(np.exp(β * smat - dmat / l), k=1)  # keep i < j

    i, j = np.nonzero(pmat)
    probas = pmat[i, j]
    probas *= nb_edges / np.sum(probas)  # normalization

    idx_edges = rng.rand(len(probas)) < probas

    if verbose:
        print(f'Nb of edges: {np.sum(idx_edges)}')

    return coords, np.squeeze(comm_vec), (i[idx_edges], j[idx_edges])


def greatcircle_distance(long1, lat1, long2, lat2, R=6371):
    """Calculate the great-circle distance between pais of points.

    Parameters
    ----------
    long, lat : float
        The longitude and latitude of the starting point.
    end_long, end_lat : float
        The longitude and latitude of the end point.
    R : float, optional
        The radius of the Earth in km (default value is 6371).

    """
    λ1, ϕ1, λ2, ϕ2 = map(np.radians, [long1, lat1, long2, lat2])
    Δϕ = ϕ1 - ϕ2  # abs value not needed because of sin squared
    Δλ = λ1 - λ2

    radical = np.sin(Δϕ / 2)**2 + np.cos(ϕ1) * np.cos(ϕ2) * np.sin(Δλ / 2)**2

    return R * 2 * np.arcsin(np.sqrt(radical))


def load_dmat(file: str):
    """
    Read distance matrix from file.

    Parameters
    ----------
    file : str
        Formats accepted: `.mat` or `.npz` (which is converted to dense matrix).
        If the matrix is upper triangular the function converts it to full.

    Returns
    -------
    np.array

    """
    assert file.endswith(('.mat', '.npz')), 'supported formats are .mat and .npz'
    format = file[-3:]

    dmat = loadmat(file)['dmat'] if format == 'mat' else sparse.load_npz(file)

    m, n = dmat.shape
    assert m == n, 'matrix should be square'

    idx_diag = np.diag_indices(n)
    assert np.all(np.isclose(dmat[idx_diag], 0.0)), \
        'diagonal elements should be zero'

    idx_tril = np.tril_indices(n, -1)
    if np.all(np.isclose(dmat[idx_tril], 0.0)):
        # TODO: default to upper triangular matrix in the other functions
        dmat = dmat + dmat.T

    if sparse.issparse(dmat):
        # sparse matrices are incompativle with masked arrays (io_matrix)
        dmat = np.asarray(dmat.todense())

    return dmat


def load_flows(file: str, zero_diag: bool = True):
    """
    Read flows from file.

    Parameters
    ----------
    file : str
        Formats accepted: `.npz`, `.csv` or `.adj` (custom-made format).

    Returns
    -------
    sparse.csr.csr_matrix

    """
    assert file.endswith(('.npz', '.csv', 'adj')), \
        'supported formats are .npz, .csv and .adj'
    format = file[-3:]

    if format == 'npz':
        out = sparse.load_npz(file)

    elif format in ('csv', 'adj'):
        df = pd.read_csv(file, header=0)
        df.columns = df.columns.str.lower()

        col_names = ['origin', 'destination', 'flow']

        if format == 'adj':
            adj_colnames_map = {'previous_store_id': 'origin',
                                'store_id': 'destination',
                                'visits': 'flow'}
            df.rename(adj_colnames_map, axis=1, inplace=True)

        idx_positive = (df.flow > 0)
        df = df.loc[idx_positive, col_names].reset_index(drop=True)

        out = sparsemat_from_flow(df, return_ids=False)
        # sorting done inside the function

    if zero_diag:
        out = sparsemat_remove_diag(out)
        # warning: changes sparsity pattern and is slow

    return out