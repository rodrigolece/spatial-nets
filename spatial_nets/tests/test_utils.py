import unittest

import numpy as np
import scipy.sparse as sp

from spatial_nets.utils import sparsity


class Testutils(unittest.TestCase):
    def test_sparsity(self):
        i = [0, 0]
        j = [0, 1]
        mat = sp.csr_matrix((j, (i, j)), shape=(2, 2))
        bmat = sparsity(mat)
        self.assertEqual(bmat.nnz, 2)
        self.assertTrue(np.all(bmat.toarray() == [[True, True], [False, False]]))
