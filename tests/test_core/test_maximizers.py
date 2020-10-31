from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from scipy.optimize.optimize import OptimizeResult

from pp.core.maximizers import InverseGaussianMaximizer
from pp.core.model import PointProcessDataset, PointProcessModel


class TestMaximizers(TestCase):
    def setUp(self) -> None:
        self.xn = np.array(
            [
                [1.000, 0.953, 1.000, 0.992, 0.984, 0.985, 0.906, 0.844, 0.875, 0.898],
                [1.000, 0.860, 0.953, 1.000, 0.992, 0.984, 0.985, 0.906, 0.844, 0.875],
                [1.000, 0.859, 0.860, 0.953, 1.000, 0.992, 0.984, 0.985, 0.906, 0.844],
                [1.000, 0.875, 0.859, 0.860, 0.953, 1.000, 0.992, 0.984, 0.985, 0.906],
                [1.000, 0.891, 0.875, 0.859, 0.860, 0.953, 1.000, 0.992, 0.984, 0.985],
                [1.000, 0.859, 0.891, 0.875, 0.859, 0.860, 0.953, 1.000, 0.992, 0.984],
            ]
        )

        self.m, self.n = self.xn.shape
        self.hasTheta0 = True
        self.p = self.n - 1
        self.wn = np.ones((self.m, 1))
        self.params = np.vstack(
            [0.5, np.ones((self.m, 1)), np.ones((self.n, 1))]
        ).squeeze(1)

    @patch("pp.core.maximizers.minimize")
    def test_inverse_gaussian_maximizer(self, minim):
        mock_res = Mock(OptimizeResult)
        mock_res.x = np.ones((self.m + self.n + 1,))
        mock_res.nit = 100
        mock_res.success = True
        minim.return_value = mock_res
        dataset = PointProcessDataset(self.xn, self.wn, self.p, self.hasTheta0)
        res = InverseGaussianMaximizer(dataset).train()
        self.assertIsInstance(res, PointProcessModel)
