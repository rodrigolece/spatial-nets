import os
import pickle
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
    assert output_file.endswith('.pkl'),  "invalid file extension; use '.pkl'"

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

    # Gravity model
    α, β, γ = locs.gravity_calibrate_all(verbose=False)
    fmat = locs.gravity_matrix(**{'γ': γ, 'α': α, 'β': β})

    pmat_prod = locs.probability_matrix(fmat, 'production')
    T_prod = locs.orig_rel[:, np.newaxis] * pmat_prod

    pmat_attrac = locs.probability_matrix(fmat, 'attraction')
    T_attrac = pmat_attrac * locs.dest_rel[np.newaxis, :]

    # Radiation model
    pmat_rad = locs.radiation_matrix(finite_correction=True)
    T_rad = locs.orig_rel[:, np.newaxis] * pmat_rad

    models = [T_prod, T_attrac, T_rad]

    cpc = [CPC(T_data, model) for model in models]
    cpl = [CPL(T_data, model) for model in models]
    rms = [NRMSE(T_data, model) for model in models]
    results = np.array([cpc, cpl, rms])

    rows = ['CPC', 'CPL', 'NMRSE']
    cols = ['', 'Grav. Prod.', 'Grav. Attr.', 'Rad.']
    tab = [[rows[i]] + results[i].tolist() for i in range(len(rows))]

    print(tabulate(tab, headers=cols, floatfmt='.3f'))
    print()

    # Calculate the number of significant edges
    models = [pmat_prod, pmat_attrac, pmat_rad]
    names = ['gravity', 'gravity', 'radiation']
    types = ['production', 'attraction', 'production']

    exact_mat = np.zeros((4, len(models)))
    approx_mat = np.zeros((4, len(models)))

    save_dict = dict()

    for k, (pmat, constr) in enumerate(tqdm(zip(models, types), total=3)):
        if args.exact:
            exact_plus, exact_minus = locs.significant_edges(
                pmat, constr, significance=sig, exact=True, verbose=False)
            n = len(exact_plus)
            nplus, nminus = np.sum(exact_plus), np.sum(exact_minus)
            exact_mat[:, k] = [nplus, nminus, n - nplus - nminus, n]
            save_dict[names[k], constr, 'exact'] = (exact_plus, exact_minus)

        approx_plus, approx_minus = locs.significant_edges(
            pmat, constr, significance=sig, exact=False, verbose=False)
        n = len(approx_plus)
        nplus, nminus = np.sum(approx_plus), np.sum(approx_minus)
        approx_mat[:, k] = [nplus, nminus, n - nplus - nminus, n]

        save_dict[names[k], constr, 'approx'] = (approx_plus, approx_minus)

    rows = ['Positive (observed larger)', 'Negative (model larger)',
            'Not-significant', 'Total']
    cols = ['', 'Grav. Prod.', 'Grav. Attr.', 'Rad.']

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
    with open(output_file, 'wb') as f:
        f.write(pickle.dumps(save_dict))

print('\nDone!')