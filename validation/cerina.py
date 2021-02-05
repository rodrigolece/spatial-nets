import os
import numpy as np
import scipy.sparse as sp

import graph_tool.all as gt

import utils
import locations


N, rho = 40, 0.1

l, beta = 0.5, 0.1
eps = 0.2
symmetric_pmat = True

coords, comm_vec, (i, j) = utils.benchmark_cerina(N, rho, l, beta, eps)

data = np.ones(len(i), dtype=int)
mat = sp.csr_matrix((data, (i,j)), shape=(N, N))  # upper triangular

cerina = utils.build_graph(mat, coords=coords, directed=False)
pruned = gt.extract_largest_component(cerina)

T_data = gt.adjacency(pruned)

locs = locations.DataLocations(cords[pruned.get_vertices()], T_data)
