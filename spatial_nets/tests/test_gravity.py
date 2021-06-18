import unittest

import numpy as np
from spatial_nets.locations import LocationsDataClass, DataNotSet
from spatial_nets.models.gravity import GravityModel


class TestGravityModel(unittest.TestCase):
    def setUp(self):
        coords = np.array([[0, 0], [0, 1], [2, 0]])  # NB 3rd point
        flow_data = np.ones((3, 3), dtype=int)
        self.locs = LocationsDataClass(flow_data, coords=coords)
        self.expected_flows = 4 * np.array(
            [
                [0, 1, 1 / 4],
                [1, 0, 1 / 5],
                [1 / 4, 1 / 5, 0],
            ],
        )

    def test_init_nomethod(self):
        with self.assertRaises(ValueError):
            GravityModel(constraint="production", method=None, coef=None)

        with self.assertRaises(ValueError):
            GravityModel(constraint="production", method=None, coef=[2, 1])

        # TODO: with invalid keys
        #  with self.assertRaises(ValueError):
        #      GravityModel(constraint="production", method=None, coef=[2, 1])

        model = GravityModel(constraint="production", method=None, coef=[2, 1, 1])
        self.assertIsNotNone(model.coef_)
        self.assertEqual(model.coef_.get("γ"), 2)
        self.assertEqual(model.coef_.get("α"), 1)
        self.assertEqual(model.coef_.get("β"), 1)

    def test_init(self):
        # maxiters should be ignored
        prod = GravityModel(constraint="production", method="linreg", maxiters=100)
        self.assertIsNotNone(prod.routine)
        self.assertIsNotNone(prod.aux_constraint)

        # TODO:                                                cpc
        attrac = GravityModel(constraint="attraction", method="nlls", maxiters=100)
        self.assertIsNotNone(attrac.routine)
        self.assertIsNotNone(attrac.aux_constraint)

        # maxiters should be used
        doubly = GravityModel(constraint="doubly", method="nlls", maxiters=100)
        self.assertIsNotNone(doubly.routine)
        self.assertIsNotNone(doubly.aux_constraint)
        self.assertEqual(doubly.aux_constraint.maxiters, 100)

    def test_fit_nnls(self):
        prod = GravityModel(constraint="production", method="nlls")
        prod.fit(self.locs)

        # test the attributes are set inside the call to fit
        self.assertEqual(prod.flow_data.sum(), 6)
        self.assertEqual(prod.aux_constraint.total_flow_, 6)

        # test the routine is working
        self.assertIsNotNone(prod.coef_)
        self.assertTrue("γ" in prod.coef_)
        self.assertEqual(prod.coef_.get("α"), 0)
        self.assertTrue("β" in prod.coef_)

    def test_fit_cpc(self):
        prod = GravityModel(constraint="production", method="cpc")
        prod.fit(self.locs)

        # test the attributes are set inside the call to fit
        self.assertEqual(prod.flow_data.sum(), 6)
        self.assertEqual(prod.aux_constraint.total_flow_, 6)

        # test the routine is working
        self.assertIsNotNone(prod.coef_)
        self.assertTrue("γ" in prod.coef_)
        self.assertEqual(prod.coef_.get("α"), 0)
        self.assertTrue("β" in prod.coef_)

    def test_fit_linreg(self):
        prod = GravityModel(constraint="production", method="linreg")
        prod.fit(self.locs)

        # test the attributes are set inside the call to fit
        self.assertEqual(prod.flow_data.sum(), 6)
        self.assertEqual(prod.aux_constraint.total_flow_, 6)

        # test the routine is working
        self.assertIsNotNone(prod.coef_)
        self.assertTrue("γ" in prod.coef_)
        self.assertEqual(prod.coef_.get("α"), 0)
        self.assertTrue("β" in prod.coef_)

    def test_transform(self):
        model = GravityModel(constraint=None, method=None, coef=[2, 1, 1])
        model.fit(self.locs)
        grav_flows = model.transform()
        self.assertTrue(np.allclose(grav_flows, self.expected_flows))
