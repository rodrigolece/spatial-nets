import os
from pathlib import Path

import numpy as np
import graph_tool.all as gt
from tqdm import tqdm

from spatial_nets import *
from spatial_nets import utils


benchmark = getattr(utils, "benchmark_expert")
directed = True
verbose = True
use_approx = False


def run_experiment(gamma, n, m, nb_repeats, nb_net_repeats, filename=None):
    N = 100

    r = np.logspace(0, 2, n)
    l = np.linspace(0.0, 2.0, m)
    rho, lamb = np.meshgrid(r, l, indexing="ij")

    # The save directories, before we modify lamb to make the network directed
    save_grav = {"rho": rho, "lamb": lamb}
    save_rad = {"rho": rho, "lamb": lamb}

    ndarray_grav = np.zeros((3, *rho.shape))  # nmi, Bs, nmi_fix
    ndarray_rad = np.zeros((3, *rho.shape))

    lamb_12 = lamb + 0.1
    lamb_21 = np.maximum(lamb - 0.1, 0.0)
    lamb = np.stack((lamb_12, lamb_21), axis=2)

    for i in range(n):
        for j in range(m):
            params = {"rho": rho[i, j], "lamb": lamb[i, j], "gamma": gamma}
            repeated_runs(
                ndarray_grav, ndarray_rad, (i, j), N, params, nb_repeats, nb_net_repeats
            )

    save_grav.update(
        {
            "nmi": ndarray_grav[0],
            "Bs": ndarray_grav[1],
            "nmif": ndarray_grav[2],
        }
    )
    save_rad.update(
        {
            "nmi": ndarray_rad[0],
            "Bs": ndarray_rad[1],
            "nmif": ndarray_rad[2],
        }
    )

    if filename is not None:
        name, ext = os.path.splitext(filename)
        print(f"\nWriting results to {name}_grav.npz")
        np.savez(f"{name}_grav.npz", **save_grav)

        print(f"\nWriting results to {name}_rad.npz")
        np.savez(f"{name}_rad.npz", **save_rad)


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
    grav = GravityModel(constraint="doubly", verbose=verbose)
    prediction = grav.fit_transform(locs)
    doubly = DoublyConstrained(approx_pvalues=use_approx, verbose=verbose)
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


def repeated_runs(
    ndarray_grav,
    ndarray_rad,
    idx,
    N,
    params,
    nb_repeats,
    nb_net_repeats,
    start_seed=0,
):
    out_grav = np.zeros((nb_repeats * nb_net_repeats, 3))
    out_rad = np.zeros_like(out_grav)

    gt.seed_rng(start_seed)

    for k in range(nb_net_repeats):
        bench, locs = benchmark_graph(N, params, seed=start_seed + k)
        grav_dc = backbone_grav_dc(locs)
        rad_pc = backbone_rad_dc(locs)
        x = bench.vp.b.a

        for i in range(nb_repeats):
            row = k * nb_repeats + i

            # gravity
            grav_dc_kwargs = {
                "layers": True,
                "state_args": {"ec": grav_dc.ep.weight, "layers": True},
            }
            state = gt.minimize_blockmodel_dl(grav_dc, **grav_dc_kwargs)
            nmi = gt.mutual_information(x, state.b.a, norm=True) * N
            B = state.get_nonempty_B()

            grav_dc_kwargs.update({"B_min": 2, "B_max": 2})
            fix = gt.minimize_blockmodel_dl(grav_dc, **grav_dc_kwargs)
            nmif = gt.mutual_information(x, fix.b.a, norm=True) * N
            #  Bf = fix.get_nonempty_B()

            #  out_grav[row] = nmi, B, state.entropy(), nmif, Bf, fix.entropy()
            out_grav[row] = nmi, B, nmif

            # radiation
            rad_pc_kwargs = {
                "layers": True,
                "state_args": {"ec": rad_pc.ep.weight, "layers": True},
            }
            state = gt.minimize_blockmodel_dl(rad_pc, **rad_pc_kwargs)
            nmi = gt.mutual_information(x, state.b.a, norm=True) * N
            B = state.get_nonempty_B()

            rad_pc_kwargs.update({"B_min": 2, "B_max": 2})
            fix = gt.minimize_blockmodel_dl(rad_pc, **rad_pc_kwargs)
            nmif = gt.mutual_information(x, fix.b.a, norm=True) * N
            #  Bf = fix.get_nonempty_B()

            #  out_rad[row] = nmi, B, state.entropy(), nmif, Bf, fix.entropy()
            out_rad[row] = nmi, B, nmif

    mn_grav = out_grav.mean(axis=0)
    mn_rad = out_grav.mean(axis=1)

    for k in range(3):
        ndarray_grav[k][idx] = mn_grav[k]
        ndarray_rad[k][idx] = mn_rad[k]

    return None


if __name__ == "__main__":
    output_dir = Path("output")
    gamma = 2.0

    n, m = 5, 5
    nb_repeats = 10
    nb_net_repeats = 10
    #  n, m = 10, 10
    #  nb_repeats = 10
    #  nb_net_repeats = 10
    filename = f"expert-rho-lamb_gamma2_{nb_repeats:02}_{nb_net_repeats:02}.npz"
    run_experiment(
        gamma, n, m, nb_repeats, nb_net_repeats, filename=output_dir / filename
    )

    # Below could be useful for tests
    #  N = 100
    #  rho = 100
    #  lamb = (0.4, 0.2)
    #  gamma = 2
    #  params = {"rho": rho, "lamb": lamb, "gamma": gamma}

    #  grav, rad = repeated_runs(N,params, 2, 1)

    #  bench, locs = benchmark_graph(N, params)
    #  grav_dc = backbone_grav_dc(locs)
    #  grav_dc_kwargs = {
    #      "layers": True,
    #      "state_args": {"ec": grav_dc.ep.weight, "layers": True},
    #  }
    #  gt.seed_rng(0)
    #  state = gt.minimize_blockmodel_dl(grav_dc, **grav_dc_kwargs)

    #  rad_pc = backbone_rad_dc(locs)
    #  rad_pc_kwargs = {
    #      "layers": True,
    #      "state_args": {"ec": rad_pc.ep.weight, "layers": True},
    #  }
    #  state = gt.minimize_blockmodel_dl(rad_pc, **rad_pc_kwargs)
    #  print(state)
