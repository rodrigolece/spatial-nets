import os
from pathlib import Path
import csv
from ast import literal_eval

import numpy as np
import graph_tool.all as gt

from spatial_nets import *
from spatial_nets import utils


benchmark = getattr(utils, "benchmark_expert")
directed = True
verbose = True
maxiters = 200
use_approx = False


def benchmark_graph(N, params, seed=0):
    coords, comm_vec, coo_mat = benchmark(N, **params, seed=seed, directed=directed)

    bench = utils.build_weighted_graph(
        coo_mat,
        directed=directed,
        coords=coords,
        vertex_properties={"b": comm_vec},
    )

    T_data = coo_mat.tocsr()
    locs = LocationsDataClass(T_data, coords=coords)

    return bench, locs


def backbone_grav_dc(locs):
    grav = GravityModel(constraint="doubly", maxiters=maxiters, verbose=verbose)
    prediction = grav.fit_transform(locs)
    doubly = DoublyConstrained(
        approx_pvalues=use_approx, maxiters=maxiters, verbose=verbose
    )
    _ = doubly.fit_transform(locs, prediction)
    pvals = doubly.pvalues()
    pvals.set_significance()

    return pvals.compute_graph()


def backbone_rad_dc(locs):
    rad = RadiationModel(finite_correction=False)
    prediction = rad.fit_transform(locs)
    prod = ProductionConstrained(approx_pvalues=use_approx, verbose=verbose)
    _ = prod.fit_transform(locs, prediction)
    pvals = prod.pvalues()
    pvals.set_significance()

    return pvals.compute_graph()


def network_fit_statistics(g, ground_truth):
    N = g.num_vertices()
    li_kwargs = {
        "layers": True,
        "state_args": {"ec": g.ep.weight, "layers": True},
    }
    state = gt.minimize_blockmodel_dl(g, **li_kwargs)
    nmi = gt.mutual_information(ground_truth, state.b.a, norm=True) * N

    # Fit with fixed B is no longer calculated to speed up computations
    #  li_kwargs.update({"B_min": 2, "B_max": 2})
    #  fix = gt.minimize_blockmodel_dl(g, **li_kwargs)

    return nmi, state.get_nonempty_B(), state.entropy()


def summarise(mat):
    mn = mat.mean(axis=0)
    std = mat.std(axis=0)
    s = mat[:, -1].argmin()
    best = mat[s, :]

    return mn, std, best


def repeated_runs(rowid, N, params, nb_repeats, nb_net_repeats, start_seed=0):
    # When params is read from a csv file it is read as string, not dict
    if isinstance(params, str):
        params = literal_eval(params)

    mat_grav = np.zeros((nb_repeats * nb_net_repeats, 3))
    mat_rad = np.zeros_like(mat_grav)

    gt.seed_rng(start_seed)

    for k in range(nb_net_repeats):
        bench, locs = benchmark_graph(N, params, seed=start_seed + k)
        grav_dc = backbone_grav_dc(locs)
        rad_pc = backbone_rad_dc(locs)
        ground_truth = bench.vp.b.a

        for i in range(nb_repeats):
            row = k * nb_repeats + i
            mat_grav[row] = network_fit_statistics(grav_dc, ground_truth)
            mat_rad[row] = network_fit_statistics(rad_pc, ground_truth)

    mn_grav, std_grav, best_grav = summarise(mat_grav)
    out_grav = [rowid] + mn_grav.tolist() + std_grav.tolist() + best_grav.tolist()

    mn_rad, std_rad, best_rad = summarise(mat_rad)
    out_rad = [rowid] + mn_rad.tolist() + std_rad.tolist() + best_rad.tolist()

    return out_grav, out_rad


if __name__ == "__main__":
    import sys
    import pandas as pd

    args_file, rowid = sys.argv[1:]
    args_file = Path(args_file)
    rowid = int(rowid)
    output_dir = Path(os.path.dirname(args_file))

    df = pd.read_csv(args_file)
    args = df.loc[df.rowid == rowid]
    row_grav, row_rad = repeated_runs(*args.iloc[0])

    with open(output_dir / f"grav_{rowid}.csv", "w") as f:
        writer = csv.writer(f, "unix")
        writer.writerow(row_grav)

    with open(output_dir / f"rad_{rowid}.csv", "w") as f:
        writer = csv.writer(f, "unix")
        writer.writerow(row_rad)
