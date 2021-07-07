import unittest
import numpy as np
import scipy.sparse as sp

from spatial_nets.locations import LocationsDataClass


class TestLocationsDataClass(unittest.TestCase):
    def setUp(self):
        self.coordinates = np.array([[0, 0], [0, 1], [1, 0]])
        #  self.distances = np.array()

    def test_int(self):
        locs = LocationsDataClass(3)
        self.assertEqual(len(locs), 3)

    def test_copy(self):
        pass

    def test_graph(self):
        pass

    def test_vectors(self):
        pass

    def test_ndarray(self):
        flow_data = np.ones((3, 3), dtype=int)
        locs = LocationsDataClass(flow_data, coords=self.coordinates)
        self.assertEqual(len(locs), 3)
        # Test flows
        self.assertEqual(
            locs.flow_data.toarray().tolist(),
            [
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ],
        )
        self.assertIsNotNone(locs.target_rows)
        self.assertIsNotNone(locs.target_cols)
        # Test dmat
        self.assertAlmostEqual(locs.dmat[1, 2], np.sqrt(2))

    def test_spmatrix(self):
        dense_flows = np.roll(np.eye(3, dtype=int), 1, axis=0)
        dense_flows[0, 0] = 2
        flow_data = sp.csr_matrix(dense_flows)
        locs = LocationsDataClass(flow_data, coords=self.coordinates)
        self.assertEqual(len(locs), 3)
        # Test flows
        self.assertEqual(
            locs.flow_data.toarray().tolist(),
            [
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
            ],
        )
        self.assertIsNotNone(locs.target_rows)
        self.assertIsNotNone(locs.target_cols)
        # Test dmat
        self.assertAlmostEqual(locs.dmat[1, 2], np.sqrt(2))
