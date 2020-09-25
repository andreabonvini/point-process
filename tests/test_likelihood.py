from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np
from scipy.optimize.optimize import OptimizeResult

from pp.likelihood import (
    compute_invgauss_negloglikel,
    compute_invgauss_negloglikel_grad,
    compute_invgauss_negloglikel_hessian,
    likel_invnorm,
)


class TestLikelInvnorm(TestCase):
    xn = np.array(
        [
            [1.0, 0.953, 1.0, 0.992, 0.984, 0.985, 0.906, 0.844, 0.875, 0.898],
            [1.0, 0.86, 0.953, 1.0, 0.992, 0.984, 0.985, 0.906, 0.844, 0.875],
            [1.0, 0.859, 0.86, 0.953, 1.0, 0.992, 0.984, 0.985, 0.906, 0.844],
            [1.0, 0.875, 0.859, 0.86, 0.953, 1.0, 0.992, 0.984, 0.985, 0.906],
            [1.0, 0.891, 0.875, 0.859, 0.86, 0.953, 1.0, 0.992, 0.984, 0.985],
            [1.0, 0.859, 0.891, 0.875, 0.859, 0.86, 0.953, 1.0, 0.992, 0.984],
        ]
    )

    m, n = xn.shape
    wn = np.ones((m, 1))
    params = np.vstack([0.5, np.ones((m, 1)), np.ones((n, 1))]).squeeze(1)

    @patch("pp.likelihood.minimize")
    def test_likel_invnorm(self, minim):
        mock_res = Mock(OptimizeResult)
        minim.return_value = mock_res
        res = likel_invnorm(self.xn, self.wn)
        assert res == mock_res

    def test_compute_invgauss_negloglikel(self):
        res = compute_invgauss_negloglikel(self.params, self.xn, self.wn)
        assert type(res) == np.float64

    def test_compute_invgauss_negloglikel_grad(self):
        res = compute_invgauss_negloglikel_grad(self.params, self.xn, self.wn)
        assert res.shape == (self.n + self.m + 1,)

    def test_compute_invgauss_negloglikel_hessian(self):
        res = compute_invgauss_negloglikel_hessian(self.params, self.xn, self.wn)
        assert res.shape == (self.n + self.m + 1, self.n + self.m + 1)
