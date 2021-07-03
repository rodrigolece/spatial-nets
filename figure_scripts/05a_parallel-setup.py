import os
from pathlib import Path
import csv

import numpy as np


def set_up_experiment(save_dir, gamma, n, m, nb_repeats, nb_net_repeats):
    N = 100

    r = np.logspace(0, 2, n)
    l = np.linspace(0.0, 2.0, m)
    rho, lamb = np.meshgrid(r, l, indexing="ij")

    # The parameters before we modify lamb to make the network directed
    save_dict = {"rho": rho, "lamb": lamb}
    np.savez(save_dir / "rho-lamb_param-grid.npz", **save_dict)

    lamb_12 = lamb + 0.1
    lamb_21 = np.maximum(lamb - 0.1, 0.0)
    lamb = np.stack((lamb_12, lamb_21), axis=2)

    with open(save_dir / "args.csv", "w") as f:
        writer = csv.writer(f, "unix")
        writer.writerow(
            [
                "rowid",
                "N",
                "params",
                "nb_repeats",
                "nb_net_repeats",
            ]
        )

        for i in range(n):
            for j in range(m):
                rowid = i * m + j
                params = {"rho": rho[i, j], "lamb": tuple(lamb[i, j]), "gamma": gamma}
                # NB: lamb[i,j] is array so it needs to be converted to tuple
                writer.writerow([rowid, N, params, nb_repeats, nb_net_repeats])

    return None


if __name__ == "__main__":
    import sys

    output_dir = Path(sys.argv[1])
    assert os.path.isdir(output_dir)

    gamma = 2.0

    n, m = 5, 4
    nb_repeats = 2
    nb_net_repeats = 2
    #  n, m = 10, 10
    #  nb_repeats = 10
    #  nb_net_repeats = 10

    #  output_dir = Path("output_data")
    set_up_experiment(output_dir, gamma, n, m, nb_repeats, nb_net_repeats)
