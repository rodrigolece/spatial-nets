import unittest

import numpy as np
import graph_tool.all as gt

class TestGraphTool(unittest.TestCase):
    def test_nmi(self):
        x = [0]*5 + [1]*5
        self.assertTrue(np.isclose(gt.mutual_information(x, x, norm=True), 1.0))

