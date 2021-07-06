import unittest

import numpy as np
import scipy.sparse as sp
from scipy.stats import norm, binom

#  import scipy.sparse as sp

from spatial_nets.base import DataNotSet
from spatial_nets.locations import LocationsDataClass
from spatial_nets.models.constraints import (
    ProductionConstrained,
    AttractionConstrained,
    #  DoublyConstrained,
)

# TODO: still need to test doubly constrained


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
        model = ProductionConstrained()  # default: approx_pvalues=False
        model.fit(self.locs).transform(self.prediction)
        self.pvals = model.pvalues()

        # Attraction: tested with transposed data
        locs_a = LocationsDataClass(flow_data.T, coords=coords)
        model = AttractionConstrained()
        model.fit(locs_a).transform(self.prediction.T)
        self.pvals_a = model.pvalues()

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
        one = binom(12, 3 / 4)
        one_pval_r = one.pmf(range(4, 13)).sum()
        one_pval_l = one.pmf(range(5)).sum()
        two = binom(12, 1 / 4)
        two_pval_r = two.pmf(range(8, 13)).sum()
        two_pval_l = two.pmf(range(9)).sum()

        # Production
        p, m = self.pvals.right, self.pvals.left
        self.assertTrue(np.allclose(p[0].toarray(), [0, one_pval_r, two_pval_r]))
        self.assertTrue(np.allclose(m[0].toarray(), [0, one_pval_l, two_pval_l]))
        self.assertEqual(p[1, 0], 3 / 4)
        self.assertEqual(m[1, 0], 3 / 4)

        # Attraction, tested with transposed data
        p, m = self.pvals_a.right, self.pvals_a.left
        self.assertTrue(np.allclose(p.toarray()[:, 0], [0, one_pval_r, two_pval_r]))
        self.assertTrue(np.allclose(m.toarray()[:, 0], [0, one_pval_l, two_pval_l]))
        self.assertEqual(p[0, 1], 3 / 4)
        self.assertEqual(m[0, 1], 3 / 4)

    def test_compute_not_significant(self):
        expected = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 0]])

        # Production
        with self.assertRaises(DataNotSet):
            self.pvals.compute_graph()
        self.pvals.set_significance()  # defauls: 0.01
        zero = self.pvals.compute_not_significant()
        self.assertTrue(sp.issparse(zero))
        self.assertTrue(np.allclose(zero.toarray().astype(int), expected))

        # Attraction, tested with transposed data
        with self.assertRaises(DataNotSet):
            self.pvals_a.compute_graph()
        self.pvals_a.set_significance()
        zero = self.pvals_a.compute_not_significant()
        self.assertTrue(sp.issparse(zero))
        self.assertTrue(np.allclose(zero.toarray().astype(int), expected.T))

    def test_compute_graph(self):
        # Production
        with self.assertRaises(DataNotSet):
            self.pvals.compute_graph()

        self.pvals.set_significance()  # defauls: 0.01
        G = self.pvals.compute_graph()
        self.assertEqual(G.num_vertices(), 3)
        self.assertEqual(G.num_edges(), 2)

        e, f = G.edges()
        self.assertEqual(e.source(), G.vertex(0))
        self.assertEqual(e.target(), G.vertex(2))
        self.assertEqual(G.ep.weight[e], 1)  # positive or right edge

        self.assertEqual(f.source(), G.vertex(0))
        self.assertEqual(f.target(), G.vertex(1))
        self.assertEqual(G.ep.weight[f], 0)  # negative or left edge

        # Attraction, tested with transposed data
        with self.assertRaises(DataNotSet):
            self.pvals_a.compute_graph()

        self.pvals_a.set_significance()  # defauls: 0.01
        G = self.pvals_a.compute_graph()
        self.assertEqual(G.num_vertices(), 3)
        self.assertEqual(G.num_edges(), 2)

        e, f = G.edges()  # for some reason this reverses the order compared to above
        self.assertEqual(f.source(), G.vertex(2))
        self.assertEqual(f.target(), G.vertex(0))
        self.assertEqual(G.ep.weight[f], 1)  # positive or right edge

        self.assertEqual(e.source(), G.vertex(1))
        self.assertEqual(e.target(), G.vertex(0))
        self.assertEqual(G.ep.weight[e], 0)  # negative or left edge
