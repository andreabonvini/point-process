from unittest import TestCase

import numpy as np

from pp.core.distributions.inverse_gaussian import likel_invgauss_consistency_check
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
