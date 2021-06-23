import unittest

import numpy as np

#  import scipy.sparse as sp

from spatial_nets.locations import LocationsDataClass
from spatial_nets.models.constraints import (
    UnconstrainedModel,
    ProductionConstrained,
    AttractionConstrained,
    DoublyConstrained,
)


class TestUnconstrainedModel(unittest.TestCase):
    def setUp(self):
        coords = np.array([[0, 0], [0, 1], [1, 0]])
        flow_data = np.ones((3, 3), dtype=int)
        self.locs = LocationsDataClass(flow_data, coords=coords)
        self.prediction = 2 * self.locs.flow_data
        # We use the input flow data as the matrix to apply the constraints to
        # TODO: change this to the gravity or rad models?

    def test_fit(self):
        model = UnconstrainedModel(constraint=None)
        model.fit(self.locs)
        self.assertEqual(model.total_flow_, 6)
        self.assertTrue(np.allclose(model.target_rows_, [2, 2, 2]))
        self.assertTrue(np.allclose(model.target_cols_, [2, 2, 2]))

    def test_transform(self):
        model = UnconstrainedModel()
        model.fit(self.locs)
        normalised_mat = model.transform(self.prediction)  # the prediction was 2*data
        self.assertEqual(normalised_mat.sum(), 6.0)
        #  self.assertIsNone(model.probabilities_)  # this model does not set the probas


class TestProductionConstrained(unittest.TestCase):
    def setUp(self):
        coords = np.array([[0, 0], [0, 1], [1, 0]])
        flow_data = np.ones((3, 3), dtype=int)
        self.locs = LocationsDataClass(flow_data, coords=coords)
        self.prediction = 2 * self.locs.flow_data
        # We use the input flow data as the matrix to apply the constraints to
        # TODO: change this to the gravity or rad models?

    def test_fit(self):
        model = ProductionConstrained()
        model.fit(self.locs)
        self.assertEqual(model.total_flow_, 6)
        self.assertTrue(np.allclose(model.target_rows_, [2, 2, 2]))
        self.assertTrue(np.allclose(model.target_cols_, [2, 2, 2]))
        self.assertIsNone(model.probabilities_)

    def test_transform(self):
        model = ProductionConstrained()
        model.fit(self.locs)
        normalised_mat = model.transform(self.prediction)  # the prediction was 2*data
        self.assertEqual(normalised_mat.sum(), 6.0)
        self.assertTrue(np.allclose(normalised_mat.sum(axis=0), self.locs.target_rows))
        self.assertIsNotNone(model.probabilities_)
        self.assertTrue(np.allclose(model.probabilities_[0], [0, 0.5, 0.5]))


class TestAttractionConstrained(unittest.TestCase):
    def setUp(self):
        coords = np.array([[0, 0], [0, 1], [1, 0]])
        flow_data = np.ones((3, 3), dtype=int)
        self.locs = LocationsDataClass(flow_data, coords=coords)
        self.prediction = 2 * self.locs.flow_data
        # We use the input flow data as the matrix to apply the constraints to
        # TODO: change this to the gravity or rad models?

    def test_fit(self):
        model = AttractionConstrained()
        model.fit(self.locs)
        self.assertEqual(model.total_flow_, 6)
        self.assertTrue(np.allclose(model.target_rows_, [2, 2, 2]))
        self.assertTrue(np.allclose(model.target_cols_, [2, 2, 2]))
        self.assertIsNone(model.probabilities_)

    def test_transform(self):
        model = AttractionConstrained()
        model.fit(self.locs)
        normalised_mat = model.transform(self.prediction)  # the prediction was 2*data
        self.assertEqual(normalised_mat.sum(), 6.0)
        self.assertTrue(np.allclose(normalised_mat.sum(axis=0), self.locs.target_cols))
        self.assertIsNotNone(model.probabilities_)
        self.assertTrue(np.allclose(model.probabilities_[:, 0], [0, 0.5, 0.5]))


class TestDoublyConstrained(unittest.TestCase):
    def setUp(self):
        coords = np.array([[0, 0], [0, 1], [1, 0]])
        flow_data = np.ones((3, 3), dtype=int)
        self.locs = LocationsDataClass(flow_data, coords=coords)
        self.prediction = 2 * self.locs.flow_data
        # We use the input flow data as the matrix to apply the constraints to
        # TODO: change this to the gravity or rad models?

    def test_fit(self):
        model = DoublyConstrained()
        model.fit(self.locs)
        self.assertEqual(model.total_flow_, 6)
        self.assertTrue(np.allclose(model.target_rows_, [2, 2, 2]))
        self.assertTrue(np.allclose(model.target_cols_, [2, 2, 2]))
        self.assertIsNone(model.probabilities_)

    def test_transform(self):
        model = DoublyConstrained()
        model.fit(self.locs)
        normalised_mat = model.transform(self.prediction)  # the prediction was 2*data
        self.assertIsNotNone(model.balancing_factors_)
        self.assertEqual(normalised_mat.sum(), 6.0)
        self.assertTrue(np.allclose(normalised_mat.sum(axis=0), self.locs.target_rows))
        self.assertTrue(np.allclose(normalised_mat.sum(axis=1), self.locs.target_cols))
        self.assertIsNotNone(model.probabilities_)
        self.assertTrue(np.allclose(model.probabilities_[0], [0, 0.5, 0.5]))

        # Below we test that we correctly factorised the probabilities as prod constrained
        self.assertTrue(
            np.allclose(
                normalised_mat, model.target_rows_[:, np.newaxis] * model.probabilities_
            )
        )
