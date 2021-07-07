import unittest

import numpy as np
import graph_tool.all as gt


class TestGraphTool(unittest.TestCase):
    def test_nmi(self):
        x = [0] * 5 + [1] * 5
        N = len(x)
        nmi = gt.mutual_information(x, x, norm=True)
        self.assertTrue(np.isclose(nmi * N, 1.0))
