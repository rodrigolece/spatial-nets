import os
from pathlib import Path
import argparse
import numpy as np
import graph_tool.all as gt
from tqdm import tqdm

from locations import DataLocations
import utils



def grav_experiment(N, rho, lamb, gamma=2.0,
                    model='gravity-doubly',
                    pvalue_constraint='production',
                    nb_repeats=10,
                    nb_net_repeats=2,
                    significance=0.01,
                    verbose=False,
                    **kwargs
                   ):

    out = np.zeros((nb_repeats * nb_net_repeats, 5))  # overlap, nmi, vi, b, entropy

    for k in range(nb_net_repeats):
        coords, comm_vec, mat = utils.benchmark_expert(N, rho, lamb, gamma=gamma, seed=k)
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
            state = gt.minimize_blockmodel_dl(graph, **kwargs)
            ov = gt.partition_overlap(x, state.b.a, norm=True)
            vi = gt.variation_information(x, state.b.a, norm=True)
            nmi = gt.mutual_information(x, state.b.a, norm=True) * N  # bug in graph-tool's nmi
            row = k*nb_repeats + i
            out[row] = ov, vi, nmi, state.get_nonempty_B(), state.entropy()

    return out


def summarise_results(mat):
    mn= mat[:,:-1].mean(axis=0)
    std = mat[:,:-1].std(axis=0)

    k = mat[:,-1].argmin()
    best = mat[k, :]

    return mn, std, best


if __name__ == '__main__':
    output_dir = Path('output') / 'benchmark_expert'

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('nb_repeats', type=int)
    parser.add_argument('nb_net_repeats', type=int)
    parser.add_argument('-n', type=int, default=20)
    parser.add_argument('-m', type=int, default=20)
    parser.add_argument('-B', '--fixB', action='store_true')

    args = parser.parse_args()

    model = args.model
    nb_repeats = args.nb_repeats
    nb_net_repeats = args.nb_net_repeats
    n, m = args.n, args.m
    fixB = args.fixB
    N = 100  # nb of nodes

    r = np.logspace(0, 2, n)
    l = np.linspace(0, 0.8, m)
    rho, lamb = np.meshgrid(r, l)

    overlap = np.zeros_like(rho)
    overlap_std = np.zeros_like(rho)
    overlap_best = np.zeros_like(rho)

    vi = np.zeros_like(rho)
    vi_std = np.zeros_like(rho)
    vi_best = np.zeros_like(rho)

    nmi = np.zeros_like(rho)
    nmi_std = np.zeros_like(rho)
    nmi_best = np.zeros_like(rho)

    Bs = np.zeros_like(rho)
    Bs_std = np.zeros_like(rho)
    Bs_best = np.zeros_like(rho)

    # entropies = np.zeros_like(rho)

    gt_kwargs = {'B_max' : 2, 'B_min': 2} if fixB else {}

    for i in tqdm(range(n)):
        for j in range(m):
            results = grav_experiment(N, rho[i,j], lamb[i,j], model=model,
                                      nb_repeats=nb_repeats,
                                      nb_net_repeats=nb_net_repeats,
                                      **gt_kwargs)
            mn, std, best = summarise_results(results)

            overlap[i,j], overlap_std[i,j], overlap_best[i,j] = mn[0], std[0], best[0]
            vi[i,j], vi_std[i,j], vi_best[i,j] = mn[1], std[1], best[1]
            nmi[i,j], nmi_std[i,j], nmi_best[i,j] = mn[2], std[2], best[2]
            Bs[i,j], Bs_std[i,j], Bs_best[i,j] = mn[3], std[3], best[3]
            # entropies[i, j] = best[-1]

    save_dict = {
        'rho': rho,
        'lamb': lamb,
        'overlap': overlap,
        'overlap_std': overlap_std,
        'vi': vi,
        'vi_std': vi_std,
        'nmi': nmi,
        'nmi_std': nmi_std,
    }

    if not fixB:
        save_dict.update({
            'Bs': Bs,
            'Bs_std': Bs_std,
        })

    if nb_net_repeats == 1:
        save_dict.update({
            'overlap_best': overlap_best,
            'vi_best': vi_best,
            'nmi_best': nmi_best,
            'Bs_best': Bs_best,
        })


    B_name = 'fixB_' if fixB else ''
    filename = f'rho-lamb_{B_name}{model}_{nb_repeats}_{nb_net_repeats}.npz'
    print(f'\nWriting results to {filename}')
    np.savez(output_dir / filename, **save_dict)

    print('Done!\n')

