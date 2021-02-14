import argparse
from pathlib import Path
import numpy as np
import graph_tool.all as gt
from tqdm import tqdm

from typing import Tuple

from spatial_nets.locations import DataLocations
from spatial_nets import utils


def experiment(
        N: int,
        rho: float,
        params: Tuple,
        model: str,
        benchmark: str = 'expert',
        nb_repeats: int = 1,
        nb_net_repeats: int = 1,
        significance: float = 0.01,
        start_seed: int = 0,
        directed: bool = False,
        verbose: bool = False,
        **kwargs
    ):

    assert benchmark in ('expert', 'cerina'), f'invalid benchmark: {benchmark}'

    out = np.zeros((nb_repeats * nb_net_repeats, 5))  # overlap, nmi, vi, B, entropy
    out_fix = np.zeros_like(out)

    fun = getattr(utils, f'benchmark_{benchmark}')
    if verbose:
        print(fun)

    for k in range(nb_net_repeats):
        coords, comm_vec, coo_mat = fun(
            N, rho, *params,
            seed=start_seed + k,
            directed=directed
        )

        bench = utils.build_weighted_graph(
            coo_mat,
            directed=directed,
            coords=coords,
            vertex_properties={'b': comm_vec}
        )

        # the ground truth
        ground_truth = gt.BlockState(bench, b=bench.vp.b)
        x = ground_truth.b.a

        # calculate the p-values to build the graph with positive edges
        T_data = coo_mat.tocsr()
        locs = DataLocations(coords, T_data)

        graph = utils.build_significant_graph(
            locs,
            model,
            sign='plus',
            coords=coords,
            significance=significance,
            verbose=verbose
        )

        if verbose:
            print(graph)

        for i in range(nb_repeats):
            row = k*nb_repeats + i

            # Varying B
            state = gt.minimize_blockmodel_dl(graph)
            ov = gt.partition_overlap(x, state.b.a, norm=True)
            vi = gt.variation_information(x, state.b.a, norm=True)
            nmi = gt.mutual_information(x, state.b.a, norm=True) * N  # bug in graph-tool's nmi
            out[row] = ov, vi, nmi, state.get_nonempty_B(), state.entropy()

            # Fixed B
            state = gt.minimize_blockmodel_dl(graph, B_max=2, B_min=2)
            ov = gt.partition_overlap(x, state.b.a, norm=True)
            vi = gt.variation_information(x, state.b.a, norm=True)
            nmi = gt.mutual_information(x, state.b.a, norm=True) * N  # bug in graph-tool's nmi
            out_fix[row] = ov, vi, nmi, state.get_nonempty_B(), state.entropy()

    return out, out_fix


def summarise_results(mat):
    mn= mat[:,:-1].mean(axis=0)
    std = mat[:,:-1].std(axis=0)

    k = mat[:,-1].argmin()
    best = mat[k, :-1]

    return mn, std, best


def main(output_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('nb_repeats', type=int)
    parser.add_argument('nb_net_repeats', type=int)
    parser.add_argument('-n', type=int, default=20)
    parser.add_argument('-m', type=int, default=20)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--directed', action='store_true')
    parser.add_argument('-s', '--globalseed', type=int, default=0)
    parser.add_argument('--nosave', action='store_true')  # for testing
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    model = args.model
    nb_repeats = args.nb_repeats
    nb_net_repeats = args.nb_net_repeats
    n, m = args.n, args.m
    gamma = args.gamma
    directed = args.directed
    N = 100  # nb of nodes

    r = np.logspace(0, 2, n)
    l = np.linspace(0.0, 1.0, m)
    rho, lamb = np.meshgrid(r, l)

    # The save directories, before we modify lamb if the network is directed
    save_dict = { 'rho': rho, 'lamb': lamb }
    save_dict_fix = { 'rho': rho, 'lamb': lamb }

    mn = [np.zeros_like(rho) for _ in range(4)]  # overlap, vi, nmi, Bs
    std = [np.zeros_like(rho) for _ in range(4)]
    best = [np.zeros_like(rho) for _ in range(4)]

    mn_fix = [np.zeros_like(rho) for _ in range(4)]  # overlap, vi, nmi, Bs
    std_fix = [np.zeros_like(rho) for _ in range(4)]
    best_fix = [np.zeros_like(rho) for _ in range(4)]

    if directed:
        lamb_12 = np.minimum(lamb + 0.1, 1.0)
        lamb_21 = np.maximum(lamb - 0.1, 0.0)
        lamb = np.stack((lamb_12, lamb_21), axis=2)

    for i in tqdm(range(n)):
        for j in range(m):
            params = (lamb[i, j], gamma)

            res, res_fix = experiment(
                N, rho[i,j], params, model,
                benchmark='expert',
                nb_repeats=nb_repeats,
                nb_net_repeats=nb_net_repeats,
                start_seed=args.globalseed,
                directed=directed,
                verbose=args.verbose
            )

            mn_res, std_res, best_res = summarise_results(res)
            mn[0][i,j], mn[1][i,j], mn[2][i,j], mn[3][i,j] = mn_res
            std[0][i,j], std[1][i,j], std[2][i,j], std[3][i,j] = std_res
            best[0][i,j], best[1][i,j], best[2][i,j], best[3][i,j] = best_res

            mn_res, std_res, best_res = summarise_results(res_fix)
            mn_fix[0][i,j], mn_fix[1][i,j], mn_fix[2][i,j], mn_fix[3][i,j] = mn_res
            std_fix[0][i,j], std_fix[1][i,j], std_fix[2][i,j], std_fix[3][i,j] = std_res
            best_fix[0][i,j], best_fix[1][i,j], best_fix[2][i,j], best_fix[3][i,j] = best_res

    save_dict.update({
        'overlap': mn[0],
        'overlap_std': std[0],
        'vi': mn[1],
        'vi_std': std[1],
        'nmi': mn[2],
        'nmi_std': std[2],
        'Bs': mn[3],
        'Bs_std': std[3]
    })

    save_dict_fix.update({
        'overlap': mn_fix[0],
        'overlap_std': std_fix[0],
        'vi': mn_fix[1],
        'vi_std': std_fix[1],
        'nmi': mn_fix[2],
        'nmi_std': std_fix[2]
    })

    if nb_net_repeats == 1:
        save_dict.update({
            'overlap_best': best[0],
            'vi_best': best[1],
            'nmi_best': best[2],
            'Bs_best': best[3]
        })

        save_dict_fix.update({
            'overlap_best': best_fix[0],
            'vi_best': best_fix[1],
            'nmi_best': best_fix[2]
        })

    if not args.nosave:
        dir_name = 'directed_' if directed else ''
        gamma_name = f'{gamma:.1f}_' if gamma != 2.0 else ''

        filename = f'{dir_name}{gamma_name}rho-lamb_{model}_{nb_repeats}_{nb_net_repeats}.npz'
        print(f'\nWriting results to {filename}')
        np.savez(output_dir / filename, **save_dict)

        filename = f'{dir_name}{gamma_name}rho-lamb_fixB_{model}_{nb_repeats}_{nb_net_repeats}.npz'
        print(f'Writing results with fixed B to {filename}')
        np.savez(output_dir / filename, **save_dict_fix)


if __name__ == '__main__':
    output_dir = Path('output_expert')
    main(output_dir)

    print('\nDone!\n')

