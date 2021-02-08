import os
from pathlib import Path
import collections
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import loadmat
from typing import List, Union
from collections.abc import Iterable

import graph_tool as gt

from sklearn.metrics import pairwise


#__all__ = [
#    'sparsemat_from_flow', 'sparsemat_remove_diag',
#    'load_dmat', 'load_flows',
#    'benchmark_cerina', 'greatcircle_distance'
#]


def _get_iterable(x):
    """Utility function."""
    if isinstance(x, collections.Iterable) and not isinstance(x, str):
        return x
    else:
        return (x,)


def build_graph(mat, idx=None, directed=True, coords=None, vertex_properties={}):
    """Build a Graph from a given mat and a subset of the nonzero entries."""
    nb_nodes, _ = mat.shape

    i, j = mat.nonzero()
    # TODO: could reuse i,j and avoid the else clause
    if idx is not None:
        ii, jj = i[idx], j[idx]
    else:
        ii, jj = i, j
    if not directed:
        pos = ii < jj
        ii, jj = ii[pos], jj[pos]

    nb_edges = len(ii)

    data = np.ones(nb_edges, dtype=int)
    A = sp.csr_matrix((data, (ii, jj)), shape=mat.shape)

    G = gt.Graph(directed=directed)
    G.add_vertex(nb_nodes)
    G.add_edge_list(np.transpose(A.nonzero()))

    if coords is not None:
        pos = G.new_vertex_property('vector<double>')
        cairo = coords.T * [[1], [-1]]
        # cairo's origin is top left and y increases downwards
        pos.set_2d_array(cairo)
        G.vertex_properties['pos'] = pos

    for name, vals in vertex_properties.items():
        if isinstance(vals[0], str):
            value_type = 'string'
        elif isinstance(vals[0], (np.floating, float)):
            value_type = 'float'
        elif isinstance(vals[0], (np.integer, int)):
            value_type = 'int'
        else:
            raise Exception('vertex_property type is not supported')

        vp = G.new_vertex_property(value_type, vals=vals)
        G.vertex_properties[name] = vp

    return G


def build_significant_graph(locs,
                            model,
                            coords=None,
                            significance=0.01,
                            verbose=False):

    family, ct = model.split('-')

    pvalue_ct = 'production' if ct == 'doubly' else ct
    # we default to production for the pvalues for the DC model

    if family == 'gravity':
        if ct == 'production':
            c, *other, b = locs.gravity_calibrate_nonlinear(constraint_type=ct)
            fmat = locs.gravity_matrix(c, α=0, β=b)

        elif ct == 'attraction':
            c, a, *other = locs.gravity_calibrate_nonlinear(constraint_type=ct)
            fmat = locs.gravity_matrix(c, α=a, β=0)

        elif ct == 'doubly':
            c, *other = locs.gravity_calibrate_nonlinear(constraint_type=ct)
            fmat = locs.gravity_matrix(c, α=0, β=0)

    elif family == 'radiation':
        fmat = locs.radiation_matrix(finite_correction=False)

    else:
        raise NotImplementedError

    T_model = locs.constrained_model(fmat, ct, verbose=verbose)
    pmat = locs.probability_matrix(T_model, pvalue_ct)
    pvals = locs.pvalues_exact(pmat, constraint_type=pvalue_ct)
    idx_plus = pvals[:, 0] < significance

    out = build_graph(locs.data, idx_plus, coords=coords, directed=True)

    return out


def build_weighted_graph(coo_mat, directed=False, coords=None, vertex_properties={}):
    """Build a weigthed Graph from COO sparse matrix."""
    assert sp.isspmatrix_coo(coo_mat), 'error: wrong matrix type'

    nb_nodes, _ = coo_mat.shape
    G = gt.Graph(directed=directed)
    G.add_vertex(nb_nodes)

    weight = G.new_edge_property('int')
    edge_list = zip(coo_mat.row, coo_mat.col, coo_mat.data)
    G.add_edge_list(edge_list, eprops=[weight])
    G.edge_properties['weight'] = weight

    if coords is not None:
        pos = G.new_vertex_property('vector<double>')
        cairo = coords.T * [[1], [-1]]
        # cairo's origin is top left and y increases downwards
        pos.set_2d_array(cairo)
        G.vertex_properties['pos'] = pos

    for name, vals in vertex_properties.items():
        if isinstance(vals[0], str):
            value_type = 'string'
        elif isinstance(vals[0], (np.floating, float)):
            value_type = 'float'
        elif isinstance(vals[0], (np.integer, int)):
            value_type = 'int'
        else:
            raise Exception('vertex_property type is not supported')

        vp = G.new_vertex_property(value_type, vals=vals)
        G.vertex_properties[name] = vp

    return G


def critical_enveloppes(locs, T_model, idx_plus, idx_minus):
    i, j = locs.data.nonzero()
    observed = np.asarray(locs.data[i, j]).flatten()
    predicted = T_model[i, j]

    plus = pd.DataFrame({'x': observed[idx_plus], 'y': predicted[idx_plus]})
    plus = plus.groupby('x')['y'].max().sort_index()
    idx = ~(plus < plus.cummax())
    top = plus.loc[idx]

    minus = pd.DataFrame({'x': observed[idx_minus], 'y': predicted[idx_minus]})
    minus = minus.groupby('x')['y'].min().sort_index()[::-1]
    idx = ~(minus > minus.cummin())
    bottom = minus.loc[idx]

    return top, bottom


def sparsemat_from_flow(flow_df, return_ids=False):
    """Convert flow DataFrame to sparse matrix."""
    idx_i, ids = pd.factorize(flow_df.origin, sort=True)
    idx_j, _ = pd.factorize(flow_df.destination, sort=True)
    data = flow_df.flow
    n = len(ids)

    mat = sp.csr_matrix((data, (idx_i, idx_j)), shape=(n, n))

    return (mat, ids) if return_ids else mat


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


# def flow_from_model(mat, epsilon, ids=None, convert_to_int=True):
#     """
#     Put output of a model in a DataFrame flow format.
#
#     epsilon is the threshold above which flows are kept.
#     """
#     n = mat.shape[0]
#
#     if ids is None:
#         ids = np.array(range(n))
#     else:
#         assert len(ids) == n, 'length of ids should match mat shape'
#
#     i, j = np.where(mat > epsilon)
#     data = mat[i, j]
#
#     if convert_to_int:
#         data = np.round(data).astype(int)
#
#     df = pd.DataFrame({'origin': ids[i], 'destination': ids[j], 'flow': data})
#
#     return df


def benchmark_cerina(nb_nodes, edge_density, l, beta, epsilon, L=1.0, seed=0):
    """Create a benchmark network of the type proposed by Cerina et al."""
    N = nb_nodes
    nb_edges = N * (N - 1) * edge_density // 2

    rng = np.random.RandomState(seed)

    # Coordinates
    ds = rng.exponential(scale=l, size=N)
    alphas = 2 * np.pi * rng.rand(N)
    shift = L * np.ones(N)
    shift[N // 2:] *= -1

    xs = ds * np.cos(alphas) + shift
    ys = ds * np.sin(alphas)
    coords = np.vstack((xs, ys)).T

    # Attibute assignment
    idx_plane = xs > 0
    idx_success = rng.rand(N) < 1 - epsilon

    comm_vec = np.zeros((N, 1), dtype=int)  # column vector
    comm_vec[np.bitwise_and(idx_plane, idx_success)] = 1
    comm_vec[np.bitwise_and(idx_plane, ~idx_success)] = -1
    comm_vec[np.bitwise_and(~idx_plane, idx_success)] = -1
    comm_vec[np.bitwise_and(~idx_plane, ~idx_success)] = 1

    # Edge selection
    smat = comm_vec.T * comm_vec
    dmat = pairwise.euclidean_distances(coords)
    pmat = np.exp(beta * smat - dmat / l)

    i, j = np.triu_indices_from(pmat, k=1)  # keep i < j
    probas = pmat[i, j]
    probas /= probas.sum()  # normalization

    draw = rng.multinomial(nb_edges, probas)
    idx, = draw.nonzero()
    mat = sp.coo_matrix((draw[idx], (i[idx], j[idx])), shape=(N, N))

    # more useful values in atrribute vector
    comm_vec[N // 2:] = 0

    # TODO: return symmetric mat
    return coords, np.squeeze(comm_vec), mat


def benchmark_expert(
        nb_nodes,
        edge_density,
        lamb,
        gamma=2,
        L=100.0,
        directed=False,
        seed=0
    ):
    """Create a benchmark network of the type proposed by Expert et al."""
    lamb = _get_iterable(lamb)
    if directed:
        assert len(lamb) == 2
    else:
        assert len(lamb) == 1, \
                'lamb should be scalar for undirected network'

    N = nb_nodes

    nb_edges = int(N * (N - 1) * edge_density )
    if not directed:
        nb_edges //= 2

    rng = np.random.RandomState(seed)

    # Coordinates
    coords = L * rng.rand(N, 2)

    # Attibute assignment
    comm_vec = np.ones(N, dtype=int)
    n = N // 2
    comm_vec[n:] = -1  # helpful for checking same/diff comm

    # Edge selection
    smat = (comm_vec * comm_vec[:, np.newaxis]).astype(float)
    # smat[smat == 1] = 1

    if directed:
        smat[:n, n:] = lamb[0]
        smat[n:, :n] = lamb[1]
    else:
        smat[smat == -1] = lamb[0]

    dmat = pairwise.euclidean_distances(coords)

    i, j = np.triu_indices_from(smat, k=1)  # i < j
    if directed:
        k, l = np.tril_indices_from(smat, k=-1)  # i > j
        i = np.concatenate((i, k))
        j = np.concatenate((j, l))

    probas = smat[i, j] / (dmat[i, j]**gamma)
    probas /= probas.sum()  # normalization

    draw = rng.multinomial(nb_edges, probas)
    idx, = draw.nonzero()
    mat = sp.coo_matrix((draw[idx], (i[idx], j[idx])), shape=(N, N))

    if not directed:
        mat = (mat + mat.T).tocoo()  # addition changes to csr

    # more useful values in atrribute vector
    comm_vec[N // 2:] = 0

    return coords, comm_vec, mat


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
    lambda1, phi1, lambda2, phi2 = map(np.radians, [long1, lat1, long2, lat2])
    Dphi = phi1 - phi2  # abs value not needed because of sin squared
    Dlambda = lambda1 - lambda2

    radical = np.sin(Dphi / 2)**2 + np.cos(phi1) * \
        np.cos(phi2) * np.sin(Dlambda / 2)**2

    return R * 2 * np.arcsin(np.sqrt(radical))


def project_mercator(longlat, R=6371):
    """Project longitude and latitute to Mercator."""
    rad = np.radians(longlat)  # lambda, phi
    x = R * rad[:,0]
    y = R * np.log(np.tan(np.pi/4 + rad[:,1]/2))

    return np.vstack((x, y)).T


def load_dmat(file: str, exclude_positions: List[int] = None):
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
    assert (ext := os.path.splitext(file)[1]) in ('.mat', '.npz'), \
            f"unsupported format: {ext} (use '.mat' or '.npz')"

    dmat = loadmat(file)['dmat'] if ext == '.mat' else sp.load_npz(file)

    m, n = dmat.shape
    assert m == n, 'matrix should be square'

    idx_diag = np.diag_indices(n)
    assert np.allclose(dmat[idx_diag], 0.0), \
        'diagonal elements should be zero'

    idx_tril = np.tril_indices(n, -1)
    if np.allclose(dmat[idx_tril], 0.0):
        # TODO: default to upper triangular matrix in the other functions
        dmat = dmat + dmat.T

    if sp.issparse(dmat):
        # sparse matrices are incompativle with masked arrays (io_matrix)
        dmat = np.asarray(dmat.todense())

    if exclude_positions is not None:
        mask = np.ones(m, dtype=bool)
        mask[exclude_positions] = False
        dmat = dmat[mask, :]
        dmat = dmat[:, mask]

    return dmat


def load_flows(file: Union[str, Path], zero_diag: bool = True):
    """
    Read flows from file.

    Parameters
    ----------
    file : str or Path
        Formats accepted: `.npz`, `.csv` or `.adj` (custom-made format).

    Returns
    -------
    sp.csr.csr_matrix

    """
    assert (ext := os.path.splitext(file)[1]) in ('.npz', '.csv', 'adj'), \
           f"unsupported format: {ext} (use '.npz', '.csv' or '.adj')"

    if ext == '.npz':
        out = sp.load_npz(file)

    elif ext in ('.csv', '.adj'):
        df = pd.read_csv(file, header=0)
        df.columns = df.columns.str.lower()

        col_names = ['origin', 'destination', 'flow']

        if ext == '.adj':
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
