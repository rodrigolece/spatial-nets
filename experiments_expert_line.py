import os
from pathlib import Path
import argparse
import numpy as np
import graph_tool.all as gt
from tqdm import tqdm

from experiments_expert import grav_experiment, summarise_results


if __name__ == '__main__':
    output_dir = Path('output') / 'benchmark_expert'

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('nb_repeats', type=int)
    parser.add_argument('nb_net_repeats', type=int)
    parser.add_argument('-m', type=int, default=20)
    parser.add_argument('-s', '--globalseed', type=int, default=0)
    # parser.add_argument('-B', '--fixB', action='store_true')

    args = parser.parse_args()

    model = args.model
    nb_repeats = args.nb_repeats
    nb_net_repeats = args.nb_net_repeats
    m = args.m
    global_seed = args.globalseed
    N = 100  # nb_of nodes
    rho = 100

    lamb = np.linspace(0, 1.0, m)

    mn = [np.zeros_like(lamb) for _ in range(4)]  # overlap, vi, nmi, Bs
    std = [np.zeros_like(lamb) for _ in range(4)]
    best = [np.zeros_like(lamb) for _ in range(4)]

    mn_fix = [np.zeros_like(lamb) for _ in range(4)]  # overlap, vi, nmi, Bs
    std_fix = [np.zeros_like(lamb) for _ in range(4)]
    best_fix = [np.zeros_like(lamb) for _ in range(4)]

    for j in tqdm(range(m)):
        res, res_fix = grav_experiment(N, rho, lamb[j], model=model,
                                       nb_repeats=nb_repeats,
                                       nb_net_repeats=nb_net_repeats,
                                       start_seed=global_seed)

        mn_res, std_res, best_res = summarise_results(res)
        mn[0][j], mn[1][j], mn[2][j], mn[3][j] = mn_res
        std[0][j], std[1][j], std[2][j], std[3][j] = std_res
        best[0][j], best[1][j], best[2][j], best[3][j] = best_res

        mn_res, std_res, best_res = summarise_results(res_fix)
        mn_fix[0][j], mn_fix[1][j], mn_fix[2][j], mn_fix[3][j] = mn_res
        std_fix[0][j], std_fix[1][j], std_fix[2][j], std_fix[3][j] = std_res
        best_fix[0][j], best_fix[1][j], best_fix[2][j], best_fix[3][j] = best_res

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

    filename = f'lamb_{model}_{nb_repeats}_{nb_net_repeats}.npz'
    print(f'\nWriting results to {filename}')
    np.savez(output_dir / filename, **save_dict)

    filename = f'lamb_fixB_{model}_{nb_repeats}_{nb_net_repeats}.npz'
    print(f'Writing results with fixed B to {filename}')
    np.savez(output_dir / filename, **save_dict_fix)

    print('\nDone!\n')

