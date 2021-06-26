import unittest

import numpy as np
from scipy.stats import norm, binom

#  import scipy.sparse as sp

from spatial_nets.base import DataNotSet
from spatial_nets.locations import LocationsDataClass
from spatial_nets.models.constraints import ProductionConstrained


class TestPvalues(unittest.TestCase):
    # I calculated the values below by hand
    def setUp(self):
        coords = np.array([[0, 0], [0, 1], [1, 0]])
        flow_data = np.array(
            [
                [0, 4, 8],
                [1, 0, 1],
                [1, 1, 0],
            ]
        )
        self.locs = LocationsDataClass(flow_data, coords=coords)
        self.prediction = np.array(
            [
                [0, 9, 3],
                [1, 0, 1],
                [1, 1, 0],
            ]
        )

    def test_approx(self):
        model = ProductionConstrained(approx_pvalues=True)
        model.fit(self.locs).transform(self.prediction)
        pvals = model.pvalues()
        p, m = pvals.right, pvals.left
        self.assertTrue(
            np.allclose(p[0].toarray(), [0, norm.cdf(10 / 3), norm.cdf(-10 / 3)])
        )
        self.assertTrue(
            np.allclose(m[0].toarray(), [0, norm.cdf(-10 / 3), norm.cdf(10 / 3)])
        )
        self.assertEqual(p[1, 0], 0.5)
        self.assertEqual(m[1, 0], 0.5)

    def test_exact(self):
        model = ProductionConstrained()  # default: approx_pvalues=False
        model.fit(self.locs).transform(self.prediction)
        pvals = model.pvalues()
        p, m = pvals.right, pvals.left
        one = binom(12, 3 / 4)
        two = binom(12, 1 / 4)
        self.assertTrue(
            np.allclose(
                p[0].toarray(),
                [0, one.pmf(range(4, 13)).sum(), two.pmf(range(8, 13)).sum()],
            )
        )
        self.assertTrue(
            np.allclose(
                m[0].toarray(),
                [0, one.pmf(range(5)).sum(), two.pmf(range(9)).sum()],
            )
        )
        self.assertEqual(p[1, 0], 3 / 4)
        self.assertEqual(m[1, 0], 3 / 4)

    def test_compute_graph(self):
        model = ProductionConstrained()  # default: approx_pvalues=False
        model.fit(self.locs).transform(self.prediction)
        pvals = model.pvalues()
        with self.assertRaises(DataNotSet):
            pvals.compute_graph()

        pvals.set_significance()  # defauls: 0.01
        G = pvals.compute_graph()
        self.assertEqual(G.num_vertices(), 3)
        self.assertEqual(G.num_edges(), 2)

        e, f = G.edges()
        self.assertEqual(e.source(), G.vertex(0))
        self.assertEqual(e.target(), G.vertex(2))
        self.assertEqual(G.ep.weight[e], 1)  # positive or right edge

        self.assertEqual(f.source(), G.vertex(0))
        self.assertEqual(f.target(), G.vertex(1))
        self.assertEqual(G.ep.weight[f], 0)  # negative or left edge
