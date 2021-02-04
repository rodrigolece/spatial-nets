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
    parser.add_argument('-m', type=int, default=50)
    parser.add_argument('-B', '--fixB', action='store_true')

    args = parser.parse_args()

    model = args.model
    nb_repeats = args.nb_repeats
    nb_net_repeats = args.nb_net_repeats
    m = args.m
    fixB = args.fixB
    N = 100  # nb_of nodes
    rho = 100

    lamb = np.linspace(0, 1.0, m)

    overlap = np.zeros_like(lamb)
    overlap_std = np.zeros_like(lamb)
    overlap_best = np.zeros_like(lamb)

    vi = np.zeros_like(lamb)
    vi_std = np.zeros_like(lamb)
    vi_best = np.zeros_like(lamb)

    nmi = np.zeros_like(lamb)
    nmi_std = np.zeros_like(lamb)
    nmi_best = np.zeros_like(lamb)

    Bs = np.zeros_like(lamb)
    Bs_std = np.zeros_like(lamb)
    Bs_best = np.zeros_like(lamb)

    # entropies = np.zeros_like(lamb)

    gt_kwargs = {'B_max' : 2, 'B_min': 2} if fixB else {}

    for j in tqdm(range(m)):
        results = grav_experiment(N, rho, lamb[j], model=model,
                                  nb_repeats=nb_repeats,
                                  nb_net_repeats=nb_net_repeats,
                                  **gt_kwargs)
        mn, std, best = summarise_results(results)

        overlap[j], overlap_std[j], overlap_best[j] = mn[0], std[0], best[0]
        vi[j], vi_std[j], vi_best[j] = mn[1], std[1], best[1]
        nmi[j], nmi_std[j], nmi_best[j] = mn[2], std[2], best[2]
        Bs[j], Bs_std[j], Bs_best[j] = mn[3], std[3], best[3]
        # entropies[i, j] = best[-1]

    save_dict = {
        'lamb': lamb,
        'overlap': overlap,
        'overlap_std': overlap_std,
        'vi': vi,
        'vi_std': vi_std,
        'nmi': nmi,
        'nmi_std': nmi_std
    }

    if not fixB:
        save_dict.update({
            'Bs': Bs,
            'Bs_std': Bs_std
        })

    if nb_net_repeats == 1:
        save_dict.update({
            'overlap_best': overlap_best,
            'vi_best': vi_best,
            'nmi_best': nmi_best,
            'Bs_best': Bs_best
        })

    B_name = 'fixB_' if fixB else ''
    filename = f'lamb_{B_name}{model}_{nb_repeats}_{nb_net_repeats}.npz'
    print(f'\nWriting results to {filename}')
    np.savez(output_dir / filename, **save_dict)

    print('Done!\n')

