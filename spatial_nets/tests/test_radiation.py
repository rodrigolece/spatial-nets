import unittest

import numpy as np
from spatial_nets.locations import LocationsDataClass
from spatial_nets.models.radiation import RadiationModel


class TestRadiationModel(unittest.TestCase):
    def setUp(self):
        coords = np.array([[0, 0], [0, 1], [2, 0]])  # NB 3rd point
        flow_data = np.ones((3, 3), dtype=int)
        self.locs = LocationsDataClass(flow_data, coords=coords)
        self.expected_flows = np.array(
            [
                [0, 1 / 2, 1 / 6],
                [1 / 2, 0, 1 / 6],
                [1 / 2, 1 / 6, 0],
            ],
        )

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            RadiationModel(threshold=-1)

    def test_fit(self):
        model = RadiationModel()  # default args
        model.fit(self.locs)
        self.assertEqual(model.N, 3)
        self.assertTrue(np.allclose(model.production, [2, 2, 2]))
        self.assertTrue(np.allclose(model.attraction, [2, 2, 2]))
        self.assertAlmostEqual(model.dmat[1, 2], np.sqrt(5))

    def test_io_matrix(self):
        # No thresholding
        model = RadiationModel(finite_correction=False)
        model.fit(self.locs)
        self.assertEqual(
            model._io_matrix(threshold=model.threshold).tolist(),
            [
                [0, 0, 2],
                [0, 0, 2],
                [0, 2, 0],
            ],
        )
        # Thresholding
        model = RadiationModel(finite_correction=False, threshold=1.0)
        model.fit(self.locs)
        mat = model._io_matrix(threshold=model.threshold)
        self.assertTrue(np.allclose(mat, np.zeros((3, 3))))

    def test_transform_nofinite(self):
        model = RadiationModel(finite_correction=False)
        model.fit(self.locs)
        rad_flows = model.transform()
        self.assertTrue(np.allclose(rad_flows, self.expected_flows))

    def test_transform_finite(self):
        model = RadiationModel(finite_correction=True)
        model.fit(self.locs)
        rad_flows = model.transform()
        self.assertTrue(np.allclose(rad_flows, 1.5 * self.expected_flows))
