import os
# import argparse
import numpy as np
from tabulate import tabulate

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
    γ, α, β = locs.gravity_calibrate_all(verbose=False)
    fmat = locs.gravity_matrix(**{'γ': γ, 'α': α, 'β': β})

    doubly_stochastic = simple_ipf(fmat, tol=1e-6)
    T_grav_prod = locs.orig_rel[:, np.newaxis] * doubly_stochastic
    T_grav_attrac = doubly_stochastic * locs.dest_rel[np.newaxis, :]

    grav_doubly = simple_ipf(fmat, locs.data_out, locs.data_in)
    # Sanity check, the two below should give same model and they do
    # pmat_prod = locs.probability_matrix(grav_doubly, 'production')
    # T_prod_const = locs.orig_rel[:, np.newaxis] * pmat_prod
    #
    # pmat_attrac = locs.probability_matrix(grav_doubly, 'attraction')
    # T_attrac_const = pmat_attrac * locs.dest_rel[np.newaxis, :]

    # Radiation model
    fmat = locs.radiation_matrix(finite_correction=False)
    doubly_stochastic = simple_ipf(fmat, tol=1e-6, maxiters=500)
    T_rad_prod = locs.orig_rel[:, np.newaxis] * doubly_stochastic
    T_rad_attrac = doubly_stochastic * locs.dest_rel[np.newaxis, :]

    rad_doubly = simple_ipf(fmat, locs.data_out, locs.data_in, maxiters=500)

    models = [T_grav_prod, T_grav_attrac, grav_doubly,
              T_rad_prod, T_rad_attrac, rad_doubly]

    cpc = [CPC(T_data, model) for model in models]
    cpl = [CPL(T_data, model) for model in models]
    rms = [RMSE(T_data, model) for model in models]
    results = np.array([cpc, cpl, rms])

    rows = ['CPC', 'CPL', 'NMRSE']
    cols = ['', 'Grav. Prod.', 'Grav. Attr.', 'Grav. Doubly',
            'Rad. Prod.', 'Rad. Attr.', 'Rad. Doubly']
    tab = [[rows[i]] + results[i].tolist() for i in range(len(rows))]

    print(tabulate(tab, headers=cols, floatfmt='.3f'))
    print('\n')  # two empty lines
