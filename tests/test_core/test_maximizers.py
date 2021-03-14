from unittest import TestCase

import numpy as np

from pp.core.maximizers import InverseGaussianMaximizer
from pp.model import InverseGaussianResult, PointProcessDataset


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

        self.n, self.m = self.xn.shape
        self.eta = np.ones((self.n, 1))
        self.hasTheta0 = True
        self.p = self.m - 1
        self.wn = np.ones((self.n, 1))
        self.params = np.vstack(
            [0.5, np.ones((self.n, 1)), np.ones((self.m, 1))]
        ).squeeze(1)
        self.current_time = 6.0
        self.xt = np.array(
            [[1.000, 0.859, 0.891, 0.875, 0.859, 0.860, 0.953, 1.000, 0.992, 0.984]]
        )
        self.target = 0.900

    def test_inverse_gaussian_maximizer(self):
        dataset = PointProcessDataset(
            self.xn, self.wn, self.p, self.eta, self.current_time, self.xt, self.target,
        )
        res = InverseGaussianMaximizer(dataset, max_steps=30).train()
        self.assertIsInstance(res, InverseGaussianResult)
