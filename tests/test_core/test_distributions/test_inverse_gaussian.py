from unittest import TestCase

import numpy as np

from pp.core.distributions.inverse_gaussian import (
    _log_inverse_gaussian,
    compute_invgauss_negloglikel,
    compute_invgauss_negloglikel_grad,
    compute_invgauss_negloglikel_hessian,
    compute_lambda,
    likel_invgauss_consistency_check,
)
from tests.data import DatasetTestData


class TestInverseGaussian(TestCase):
    def setUp(self) -> None:
        data = DatasetTestData()
        self.inter_events_times = data.inter_events_times
        self.p = data.p
        self.xn = data.xn
        self.n, self.m = data.xn.shape
        self.wn = data.wn
        self.k = data.k
        self.theta = data.theta
        self.eta = data.eta
        self.mus = data.mus
        self.params = data.params

    def test_likel_invgauss_consistency_check_good(self):
        res = likel_invgauss_consistency_check(
            xn=self.xn, wn=self.wn, xt=None, thetap0=self.theta
        )
        self.assertIsNone(res)

    def test_likel_invgauss_consistency_check_wn_bad(self):
        with self.assertRaises(ValueError):
            likel_invgauss_consistency_check(
                xn=self.xn, wn=np.ones((self.n - 1, 1)), xt=None, thetap0=self.theta,
            )

    def test_likel_invgauss_consistency_check_xt_bad(self):
        with self.assertRaises(ValueError):
            likel_invgauss_consistency_check(
                xn=self.xn, wn=self.wn, xt=np.ones((1, self.n - 1)), thetap0=self.theta,
            )

    def test_likel_invgauss_consistency_check_thetap_bad(self):
        with self.assertRaises(ValueError):
            likel_invgauss_consistency_check(
                xn=self.xn, wn=self.wn, xt=None, thetap0=np.ones((self.n - 1, 1)),
            )

    def test_compute_invgauss_negloglikel(self):
        res = compute_invgauss_negloglikel(self.params, self.xn, self.wn, self.eta)
        assert type(res) == np.float64

    def test_compute_invgauss_negloglikel_grad(self):
        res = compute_invgauss_negloglikel_grad(self.params, self.xn, self.wn, self.eta)
        assert res.shape == (self.m + 1,)

    def test_compute_invgauss_negloglikel_hessian(self):
        res = compute_invgauss_negloglikel_hessian(
            self.params, self.xn, self.wn, self.eta
        )
        assert res.shape == (self.m + 1, self.m + 1)

    def test_log_inverse_gaussian_unequalshapes(self):
        xs = np.array([[1], [2], [3]])
        mus = np.array([[1], [2]])
        with self.assertRaises(ValueError):
            _log_inverse_gaussian(xs, mus, lamb=500.0)

    def test_log_inverse_gaussian_wrongtypes(self):
        xs = np.array([[1], [2], [3]])
        mus = 1.0
        with self.assertRaises(TypeError):
            _log_inverse_gaussian(xs, mus, lamb=500.0)

    def test_compute_lambda(self):
        res = compute_lambda(mu=1.0, k=1700.0, time=0.85)
        self.assertGreater(res, 0.0)
