import os
# import argparse
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from locations import *
import utils

# parser = argparse.ArgumentParser(description='Parse input files')
# parser.add_argument('flow_file', help='The name of the file containing flows')
# parser.add_argument('dmat_file', help='The file constaining the distance matrix')
# args = parser.parse_args()

flow_file = os.path.join('data', 'UK_commute2011.npz')

dmat_files = [
    os.path.join('data', x)
    for x in ['UK_here_dmat.npz', 'UK_geodesic_dmat.mat']
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
    types = ['production', 'attraction', 'production']

    exact_mat = np.zeros((3, len(models)))
    approx_mat = np.zeros((3, len(models)))

    for k, (pmat, constr) in enumerate(tqdm(zip(models, types), total=3)):
        _, exact_counts = locs.significant_exact(
            pmat, constr, return_counts=True)
        exact_mat[:, k] = exact_counts
        _, approx_counts = locs.significant_approx(
            pmat, constr, return_counts=True)
        approx_mat[:, k] = approx_counts

    rows = ['Positive (observed larger)', 'Negative (model larger)',
            'Not-significant:']
    cols = ['', 'Grav. Prod.', 'Grav. Attr.', 'Rad.']

    print('\nExact p-value')
    tab = [[rows[k]] + exact_mat[k].tolist() for k in range(len(rows))]
    print(tabulate(tab, headers=cols))

    print('\nNormal approximation')
    tab = [[rows[k]] + approx_mat[k].tolist() for k in range(len(rows))]
    print(tabulate(tab, headers=cols))

    print('\n')  # two empty lines
