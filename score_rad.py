import os
# import pickle
import argparse
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from locations import *
import utils

parser = argparse.ArgumentParser(description='Parse input files')
# parser.add_argument('flow_file', help='The name of the file containing flows')
# parser.add_argument('dmat_file', help='The file constaining the distance matrix')
parser.add_argument('-e', '--exact', action='store_true',
                    help='Use exact calculation of p-values')
parser.add_argument('-o', '--output', default=None,
                    help='Name of output file for incdices')
parser.add_argument('-s', '--significance', type=float, default=0.01,
                    help='Significance threshold')

args = parser.parse_args()

sig = args.significance
output_file = args.output
if output_file:
    # assert output_file.endswith('.pkl'),  "invalid file extension; use '.pkl'"
    assert output_file.endswith('.npz'),  "invalid file extension; use '.npz'"

flow_file = os.path.join('data', 'UK_commute2011.npz')

dmat_files = [
    os.path.join('data', x)
    for x in ['UK_geodesic_dmat.mat', 'UK_here_dmat.npz']
]

print(f"\nReading flows from '{flow_file}'")

T_data = utils.load_flows(flow_file, zero_diag=True)
nb_locations, nb_pairwise_flows = T_data.shape[0], T_data.nnz

print(f'\nNb of locations loaded: {nb_locations}')
print(f'Density of flow data: {nb_pairwise_flows / nb_locations**2:.3f}')


for dmat in dmat_files:
    print(f"\nReading distance matrix from '{os.path.basename(dmat)}'\n")
    dmat = utils.load_dmat(dmat)

    locs = DataLocations(dmat, T_data)

    # Radiation: naturally production
    pmat_prod = locs.radiation_matrix(finite_correction=True)
    T_prod = locs.data_out[:, np.newaxis] * pmat_prod

    # Radiation: unconstrained
    K = locs.data.sum() / pmat_prod.sum()
    T_unc = K * pmat_prod

    # Radiation: attraction
    pmat_attrac = locs.probability_matrix(T_prod, 'attraction')
    T_attrac = pmat_attrac * locs.data_in[np.newaxis, :]

    # Radiation: doubly
    T_doubly = simple_ipf(pmat_prod, locs.data_out, locs.data_in,
                          maxiters=500, verbose=True)
    pmat_doubly_prod = locs.probability_matrix(T_doubly, 'production')
    pmat_doubly_attrac = locs.probability_matrix(T_doubly, 'attraction')

    models = [T_unc, T_prod, T_attrac, T_doubly]

    cpc = [CPC(T_data, model) for model in models]
    cpl = [CPL(T_data, model) for model in models]
    rms = [NRMSE(T_data, model) for model in models]
    results = np.array([cpc, cpl, rms])

    rows = ['CPC', 'CPL', 'NMRSE']
    cols = ['', 'Rad. Unc.', 'Rad. Prod.', 'Rad. Attr.', 'Rad. Doubly']
    tab = [[rows[i]] + results[i].tolist() for i in range(len(rows))]

    print(tabulate(tab, headers=cols, floatfmt='.3f'))
    print()

    # Calculate the number of significant edges
    models = [pmat_prod, pmat_attrac, pmat_doubly_prod, pmat_doubly_attrac]
    names = ['radiation', 'radiation', 'rad-doubly-p', 'rad-doubly-a']
    types = ['production', 'attraction', 'production', 'attraction']

    exact_mat = np.zeros((4, len(models)))
    approx_mat = np.zeros((4, len(models)))

    save_dict = dict()

    for k, (pmat, constr) in enumerate(tqdm(zip(models, types), total=len(names))):
        if args.exact:
            # exact_plus, exact_minus = locs.significant_edges(
            #     pmat, constr, significance=sig, exact=True, verbose=False)
            pvals = locs.pvalues_exact(pmat, constraint_type=constr)
            significant = pvals < sig
            exact_plus, exact_minus = significant[:, 0], significant[:, 1]

            n = len(exact_plus)
            nplus, nminus = np.sum(exact_plus), np.sum(exact_minus)
            exact_mat[:, k] = [nplus, nminus, n - nplus - nminus, n]

            name = f'{names[k]}_{constr}_exact'
            save_dict[name] = pvals.copy()  # copy maybe not needed

        # approx_plus, approx_minus = locs.significant_edges(
        #     pmat, constr, significance=sig, exact=False, verbose=False)
        pvals = locs.pvalues_approx(pmat, constraint_type=constr)
        significant = pvals < sig
        approx_plus, approx_minus = significant[:, 0], significant[:, 1]

        n = len(approx_plus)
        nplus, nminus = np.sum(approx_plus), np.sum(approx_minus)
        approx_mat[:, k] = [nplus, nminus, n - nplus - nminus, n]

        name = f'{names[k]}_{constr}_approx'
        save_dict[name] = pvals.copy()  # copy maybe not needed

    rows = ['Positive (observed larger)', 'Negative (model larger)',
            'Not-significant', 'Total']
    cols = ['', 'Rad. Prod.', 'Rad. Attr.', 'Rad. Dp', 'Rad. Da']

    if args.exact:
        print('\nExact p-value')
        tab = [[rows[k]] + exact_mat[k].tolist() for k in range(len(rows))]
        print(tabulate(tab, headers=cols))

    print('\nNormal approximation')
    tab = [[rows[k]] + approx_mat[k].tolist() for k in range(len(rows))]
    print(tabulate(tab, headers=cols))

    print('\n')  # two empty lines

# Save to file
# output_file = 'output/here_indices.pkl'

if output_file:
    print(f'Saving file: {output_file}')
    np.savez(output_file, **save_dict)
    # with open(output_file, 'wb') as f:
    #     f.write(pickle.dumps(save_dict))

print('\nDone!')
