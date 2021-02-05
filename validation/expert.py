import argparse
from pathlib import Path
import numpy as np
import graph_tool.all as gt
from tqdm import tqdm

from spatial_nets.models.locations import DataLocations
from spatial_nets.models import utils


def grav_experiment(N, rho, lamb, gamma=2.0,
                    model='gravity-doubly',
                    pvalue_constraint='production',
                    nb_repeats=10,
                    nb_net_repeats=2,
                    significance=0.01,
                    start_seed=0,
                    verbose=False
                   ):

    out = np.zeros((nb_repeats * nb_net_repeats, 5))  # overlap, nmi, vi, b, entropy
    out_fix = np.zeros_like(out)

    for k in range(nb_net_repeats):
        coords, comm_vec, mat = utils.benchmark_expert(N, rho, lamb,
                                                       gamma=gamma,
                                                       seed=start_seed + k)
        bench = utils.build_weighted_graph(mat,
                                           coords=coords,
                                           vertex_properties={'b': comm_vec},
                                           directed=False)  # the default

        t_data = gt.adjacency(bench, weight=bench.ep.weight).astype(int)
        locs = DataLocations(coords, t_data)

        # calculate the p-values to build the graph with positive edges
        graph = utils.build_significant_graph(locs, coords,
                                              significance=significance,
                                              verbose=verbose)

        # the ground truth
        ground_truth = gt.BlockState(bench, b=bench.vp.b)
        x = ground_truth.b.a

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
    parser.add_argument('-s', '--globalseed', type=int, default=0)
    parser.add_argument('--nosave', action='store_true')  # for testing
    # parser.add_argument('-B', '--fixB', action='store_true')

    args = parser.parse_args()

    model = args.model
    nb_repeats = args.nb_repeats
    nb_net_repeats = args.nb_net_repeats
    n, m = args.n, args.m
    global_seed = args.globalseed
    N = 100  # nb of nodes

    r = np.logspace(0, 2, n)
    l = np.linspace(0.0, 1.0, m)
    rho, lamb = np.meshgrid(r, l)

    mn = [np.zeros_like(rho) for _ in range(4)]  # overlap, vi, nmi, Bs
    std = [np.zeros_like(rho) for _ in range(4)]
    best = [np.zeros_like(rho) for _ in range(4)]

    mn_fix = [np.zeros_like(rho) for _ in range(4)]  # overlap, vi, nmi, Bs
    std_fix = [np.zeros_like(rho) for _ in range(4)]
    best_fix = [np.zeros_like(rho) for _ in range(4)]

    for i in tqdm(range(n)):
        for j in range(m):
            res, res_fix = grav_experiment(N, rho[i,j], lamb[i,j], model=model,
                                           nb_repeats=nb_repeats,
                                           nb_net_repeats=nb_net_repeats,
                                           start_seed=global_seed)

            mn_res, std_res, best_res = summarise_results(res)
            mn[0][i,j], mn[1][i,j], mn[2][i,j], mn[3][i,j] = mn_res
            std[0][i,j], std[1][i,j], std[2][i,j], std[3][i,j] = std_res
            best[0][i,j], best[1][i,j], best[2][i,j], best[3][i,j] = best_res

            mn_res, std_res, best_res = summarise_results(res_fix)
            mn_fix[0][i,j], mn_fix[1][i,j], mn_fix[2][i,j], mn_fix[3][i,j] = mn_res
            std_fix[0][i,j], std_fix[1][i,j], std_fix[2][i,j], std_fix[3][i,j] = std_res
            best_fix[0][i,j], best_fix[1][i,j], best_fix[2][i,j], best_fix[3][i,j] = best_res

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

    save_dict_fix = {
        'rho': rho,
        'lamb': lamb,
        'overlap': mn_fix[0],
        'overlap_std': std_fix[0],
        'vi': mn_fix[1],
        'vi_std': std_fix[1],
        'nmi': mn_fix[2],
        'nmi_std': std_fix[2]
    }

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
        filename = f'rho-lamb_{model}_{nb_repeats}_{nb_net_repeats}.npz'
        print(f'\nWriting results to {filename}')
        np.savez(output_dir / filename, **save_dict)

        filename = f'rho-lamb_fixB_{model}_{nb_repeats}_{nb_net_repeats}.npz'
        print(f'Writing results with fixed B to {filename}')
        np.savez(output_dir / filename, **save_dict_fix)

    print('\nDone!\n')


if __name__ == '__main__':
    output_dir = Path('output_expert')
    main(output_dir)

