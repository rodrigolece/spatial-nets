import argparse
from pathlib import Path
import numpy as np
import graph_tool.all as gt
from tqdm import tqdm

from expert import experiment, summarise_results


def main(output_dir):

    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('nb_repeats', type=int)
    parser.add_argument('nb_net_repeats', type=int)
    parser.add_argument('-m', type=int, default=20)
    parser.add_argument('--ell', type=float, default=1.0)
    parser.add_argument('--epsilon', type=float, default=0.0)
    parser.add_argument('--directed', action='store_true')
    parser.add_argument('-s', '--globalseed', type=int, default=0)
    parser.add_argument('--nosave', action='store_true')  # for testing
    parser.add_argument('-v', '--verbose', action='store_true')

    args = parser.parse_args()

    model = args.model
    nb_repeats = args.nb_repeats
    nb_net_repeats = args.nb_net_repeats
    m = args.m
    ell = args.ell
    epsilon = args.epsilon
    directed = args.directed
    N = 100  # nb_of nodes
    rho = 100

    beta = np.logspace(-1, 2, m)

    # The save directories
    save_dict = { 'beta': beta }
    save_dict_fix = { 'beta': beta }

    mn = [np.zeros_like(beta) for _ in range(4)]  # overlap, vi, nmi, Bs
    std = [np.zeros_like(beta) for _ in range(4)]
    best = [np.zeros_like(beta) for _ in range(4)]

    mn_fix = [np.zeros_like(beta) for _ in range(4)]  # overlap, vi, nmi, Bs
    std_fix = [np.zeros_like(beta) for _ in range(4)]
    best_fix = [np.zeros_like(beta) for _ in range(4)]

    for j in tqdm(range(m)):
        params = (ell, beta[j], epsilon)

        res, res_fix = experiment(
            N, rho, params, model,
            benchmark='cerina',
            nb_repeats=nb_repeats,
            nb_net_repeats=nb_net_repeats,
            start_seed=args.globalseed,
            directed=directed,
            verbose=args.verbose
        )

        mn_res, std_res, best_res = summarise_results(res)
        mn[0][j], mn[1][j], mn[2][j], mn[3][j] = mn_res
        std[0][j], std[1][j], std[2][j], std[3][j] = std_res
        best[0][j], best[1][j], best[2][j], best[3][j] = best_res

        mn_res, std_res, best_res = summarise_results(res_fix)
        mn_fix[0][j], mn_fix[1][j], mn_fix[2][j], mn_fix[3][j] = mn_res
        std_fix[0][j], std_fix[1][j], std_fix[2][j], std_fix[3][j] = std_res
        best_fix[0][j], best_fix[1][j], best_fix[2][j], best_fix[3][j] = best_res

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
        params_name = f'{ell:.1f}-{epsilon:.1f}_'

        filename = f'{dir_name}{params_name}beta_{model}_{nb_repeats}_{nb_net_repeats}.npz'
        print(f'\nWriting results to {filename}')
        np.savez(output_dir / filename, **save_dict)

        filename = f'{dir_name}{params_name}beta_fixB_{model}_{nb_repeats}_{nb_net_repeats}.npz'
        print(f'Writing results with fixed B to {filename}')
        np.savez(output_dir / filename, **save_dict_fix)

    print('\nDone!\n')



if __name__ == '__main__':
    output_dir = Path('output_cerina')
    main(output_dir)
