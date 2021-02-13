import argparse
from pathlib import Path
import numpy as np
# import scipy.sparse as sp
from sklearn.metrics import pairwise
import graph_tool.all as gt
from tqdm import tqdm
from pymatbridge import Matlab

from typing import Tuple

from spatial_nets import utils


def start_matlab(load_dir):
    mlab = Matlab()
    mlab.start()

    status = mlab.run_code(f'addpath {load_dir.absolute()}')['success']
    assert status, 'Error loading libraries'

    return mlab


def modularity_spa_GN(flow_mat, dist_mat, N, binsize):
    """
    Originally by Paul Expert.

    Arguments
    ---------
    flow_mat : adjacency matrix
    dist_mat : distance matrix between the nodes
    N : a measure of te importance of a node (by defaults its strength: Dist=sum(flow_mat,1) for example)
    binsize : size of the bins in the estimation of the deterrence function (has to be tuned)

    Returns
    -------
    modularity_Spa
    modularity_GN
    deterrence_fct
    """

    nb_nodes, _ = flow_mat.shape
    nbins = 2000  # nb of bins

    deterrence_fct = np.zeros(nbins)
    norma_deterrence = np.zeros(nbins)

    matrix_distance = np.zeros((nb_nodes, nb_nodes), dtype=int)
    # null_model_GN = np.zeros_like(matrix_distance)
    null_model_Spa = np.zeros((nb_nodes, nb_nodes))

    # flow_mat = flow_mat + flow_mat.T  # symmetrised matrix (doesn't change the outcome of comm. detect: arXiv:0812.1770)
    degree = flow_mat.sum(axis=0)
    null_N = N[:, np.newaxis] * N
    matrix = flow_mat / null_N  # normalised adjacency matrix

    for i in range(nb_nodes):
        for j in range(nb_nodes):

            # convert distances in binsize's units
            dist = int(np.ceil(dist_mat[i,j] / binsize))
            matrix_distance[i, j] = dist

            # weighted average for the deterrence function
            num = matrix[i, j]
            deterrence_fct[dist] += num * N[i] * N[j]
            norma_deterrence[dist] += N[i] * N[j]

    # normalisation of the deterrence function
    for i in range(nbins):
        if norma_deterrence[i] > 0:
            deterrence_fct[i] /= norma_deterrence[i]

    # computation of the randomised correlations (preserving space), spatial null-model
    for i in range(nb_nodes):
        for j in range(nb_nodes):
            null_model_Spa[i, j] = deterrence_fct[matrix_distance[i, j]]

    # the modularity matrix for the spatial null-model
    prod = null_N * null_model_Spa
    modularity_Spa = flow_mat - prod * flow_mat.sum() / prod.sum()

    # the modularity matrix for the GN null-model
    null_model_GN = degree[:, np.newaxis] * degree / degree.sum()
    modularity_GN = flow_mat - null_model_GN

    return modularity_Spa, modularity_GN, deterrence_fct


def wrapper_spectral23(mlab, A, B):
    res = mlab.run_func('callSpectral23.m', {'A': A, 'B': B}, nargout=2)

    if res['success']:
        S, Q = res['result']
        Q = Q.flatten()
        best = np.argmax(Q)
        out = S[best,:], Q[best] / A.sum()
    else:
        out = None

    return out


def experiment(
        N: int,
        rho: float,
        params: Tuple,
        benchmark: str = 'expert',
        nb_repeats: int = 1,
        nb_net_repeats: int = 1,
        start_seed: int = 0,
        verbose: bool = False,
        **kwargs
        # directed=False
    ):
    assert benchmark in ('expert', 'cerina'), f'invalid benchmark: {benchmark}'

    out = np.zeros((nb_repeats * nb_net_repeats, 5))  # overlap, nmi, vi, B, Q
    fun = getattr(utils, f'benchmark_{benchmark}')

    if verbose:
        print(fun)

    for k in range(nb_net_repeats):
        coords, comm_vec, coo_mat = fun(
            N, rho, *params,
            seed=start_seed + k,
            directed=False
        )

        bench = utils.build_weighted_graph(
            coo_mat,
            directed=False,
            coords=coords,
            vertex_properties={'b': comm_vec}
        )

        # the ground truth
        ground_truth = gt.BlockState(bench, b=bench.vp.b)
        x = ground_truth.b.a

        T_data = coo_mat.toarray()  # .tocsr()
        O_vec = T_data.sum(axis=1)
        dmat = pairwise.euclidean_distances(coords)

        for i in range(nb_repeats):
            row = k*nb_repeats + i

            spa, _, _ = modularity_spa_GN(T_data, dmat, O_vec, kwargs['binsize'])
            y, Q = wrapper_spectral23(kwargs['mlab'], T_data, spa)

            if y is not None:
                ov = gt.partition_overlap(x, y, norm=True)
                vi = gt.variation_information(x, y, norm=True)
                nmi = gt.mutual_information(x, y, norm=True)
                B = len(np.unique(y))
                out[row] = ov, vi, nmi, B, Q

    return out


def summarise_results(mat):
    mn= mat[:,:-1].mean(axis=0)
    std = mat[:,:-1].std(axis=0)

    k = mat[:,-1].argmin()
    best = mat[k, :-1]

    return mn, std, best


def main(mlab, output_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('nb_repeats', type=int)
    parser.add_argument('nb_net_repeats', type=int)
    parser.add_argument('-n', type=int, default=20)
    parser.add_argument('-m', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('-s', '--globalseed', type=int, default=0)
    parser.add_argument('--binsize', type=float, default=2.0)
    parser.add_argument('--nosave', action='store_true')  # for testing
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    nb_repeats = args.nb_repeats
    nb_net_repeats = args.nb_net_repeats
    n, m = args.n, args.m
    gamma = args.gamma
    binsize = args.binsize
    N = 100  # nb of nodes

    r = np.logspace(0, 2, n)
    l = np.linspace(0.0, 1.0, m)
    rho, lamb = np.meshgrid(r, l)

    mn = [np.zeros_like(rho) for _ in range(4)]  # overlap, vi, nmi, Bs
    std = [np.zeros_like(rho) for _ in range(4)]
    best = [np.zeros_like(rho) for _ in range(4)]

    modularity_kwargs = {'mlab': mlab, 'binsize': binsize}

    for i in tqdm(range(n)):
        for j in range(m):
            params = (lamb[i, j], gamma)

            res = experiment(
                N, rho[i, j], params,
                benchmark='expert',
                nb_repeats=nb_repeats,
                nb_net_repeats=nb_net_repeats,
                start_seed=args.globalseed,
                verbose=args.verbose,
                **modularity_kwargs
            )

            mn_res, std_res, best_res = summarise_results(res)
            mn[0][i,j], mn[1][i,j], mn[2][i,j], mn[3][i,j] = mn_res
            std[0][i,j], std[1][i,j], std[2][i,j], std[3][i,j] = std_res
            best[0][i,j], best[1][i,j], best[2][i,j], best[3][i,j] = best_res

    save_dict = {
        'rho': rho,
        'lamb': lamb,
        'overlap': mn[0],
        'overlap_std': std[0],
        'vi': mn[1],
        'vi_std': std[1],
        'nmi': mn[2],
        'nmi_std': std[2],
        'Bs': mn[3],
        'Bs_std': std[3]
    }

    if nb_net_repeats == 1:
        save_dict.update({
            'overlap_best': best[0],
            'vi_best': best[1],
            'nmi_best': best[2],
            'Bs_best': best[3]
        })

    if not args.nosave:
        gamma_name = f'{gamma:.1f}_' if gamma != 2.0 else ''

        filename = f'modularity_{binsize}_{gamma_name}rho-lamb_{nb_repeats}_{nb_net_repeats}.npz'
        print(f'\nWriting results to {filename}')
        np.savez(output_dir / filename, **save_dict)


if __name__ == '__main__':
    output_dir = Path('output_modularity')
    mlab = start_matlab(Path('.'))
    main(mlab, output_dir)
    mlab.stop()

    print('\nDone!\n')

    # data_dir = Path('output_expert')
    # flow_mat = sp.load_npz(data_dir / 'test_flow.npz').toarray()
    # N = flow_mat.sum(axis=1)
    # coords = np.load(data_dir / 'test_coords.npy')
    # dmat = pairwise.euclidean_distances(coords)

    # binsize = 2

    # spa, gn, det_fct = modularity_spa_GN(flow_mat, dmat, N, binsize)
    # print(np.round(spa[:3,:3], 4))
    # print(np.round(gn[:3,:3], 4))
    # print(np.round(det_fct[60:70], 4))

    # mlab = start_matlab(Path('.'))

    # res = mlab.run_func('callSpectral23.m', {'A': flow_mat, 'B': gn}, nargout=2)

    # if res['success']:
    #     S, Q = res['result']
    #     Q = Q.copy() / flow_mat.sum()
    #     print(Q)
    #     print(S[0,:])

    # mlab.stop()

