from unittest import TestCase

import numpy as np

from data import InverseGaussianTestData
from pp.core.distributions.inverse_gaussian import (
    build_ig_model,
    compute_invgauss_negloglikel,
    compute_invgauss_negloglikel_grad,
    compute_invgauss_negloglikel_hessian,
    inverse_gaussian,
    likel_invgauss_consistency_check,
)


class TestInverseGaussian(TestCase):
    def setUp(self) -> None:
        data = InverseGaussianTestData()
        self.inter_events_times = data.inter_events_times
        self.p = data.p
        self.xn = data.xn
        self.n, _ = data.xn.shape
        self.wn = data.wn
        self.k = data.k
        self.theta = data.theta
        self.eta = data.eta
        self.mus = data.mus
        self.params = data.params

    def test_inverse_gaussian_floats(self):
        res = inverse_gaussian(xs=2.0, mus=2.0, lamb=1.0)
        assert 1 > res > 0

    def test_inverse_gaussian_vectors(self):
        xs = np.array([1, 2, 3, 4, 5]).astype(float).reshape((5, 1))
        mus = np.array([1, 2, 3, 4, 5]).astype(float).reshape((5, 1))
        res = inverse_gaussian(xs=xs, mus=mus, lamb=1.0)
        assert res.shape == (5, 1) and all(1 > p > 0 for p in res)

    def test_inverse_gaussian_value_error(self):
        xs = np.array([1, 2, 3, 4, 5]).astype(float).reshape((5, 1))
        mus = np.array([1, 2, 3, 4]).astype(float).reshape((4, 1))
        with self.assertRaises(ValueError):
            inverse_gaussian(xs=xs, mus=mus, lamb=1.0)

    def test_inverse_gaussian_type_error(self):
        xs = np.array([1, 2, 3, 4, 5]).astype(float).reshape((5, 1))
        mus = 0.5
        with self.assertRaises(TypeError):
            inverse_gaussian(xs=xs, mus=mus, lamb=1.0)

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

    def test_build_ig_model_hasTheta0_true(self):
        hasTheta0 = True
        ar_order = 3
        theta = np.ones(ar_order + 1) / (ar_order + 1)
        k = 1.00
        results = [0.8, 0.3, 0.1]
        params_history = [theta, theta, theta]
        pp_model = build_ig_model(theta, k, hasTheta0, results, params_history)
        inter_event_times = np.ones((ar_order,))
        res = pp_model(inter_event_times)
        self.assertEqual(res.mu, 1.00)
        self.assertEqual(res.sigma, 1.00)

    def test_ig_model_wrong_usage(self):
        hasTheta0 = True
        ar_order = 3
        theta = np.ones(ar_order + 1) / (ar_order + 1)
        k = 1.00
        results = [0.8, 0.3, 0.1]
        params_history = [theta, theta, theta]
        pp_model = build_ig_model(theta, k, hasTheta0, results, params_history)
        wrong_shape_inter_event_times = np.ones((ar_order + 1,))
        with self.assertRaises(ValueError):
            pp_model(wrong_shape_inter_event_times)

    def test_compute_invgauss_negloglikel(self):
        res = compute_invgauss_negloglikel(self.params, self.xn, self.wn, self.eta)
        assert type(res) == np.float64

    # def test_compute_invgauss_k_constraint_unsatisfied(self):
    #    bad_k = -50
    #    bad_params = deepcopy(self.params)
    #    bad_params[0] = bad_k
    #    with self.assertRaises(Exception):
    #        compute_invgauss_negloglikel(bad_params, self.xn, self.wn, self.eta)

    # def test_compute_invgauss_theta_constraint_unsatisfied(self):
    #     bad_params = deepcopy(self.params)
    #    bad_params[1:] = -self.params[1:]
    #    with self.assertRaises(Exception):
    #        compute_invgauss_negloglikel(bad_params, self.xn, self.wn, self.eta)

    def test_compute_invgauss_negloglikel_grad(self):
        res = compute_invgauss_negloglikel_grad(self.params, self.xn, self.wn, self.eta)
        assert res.shape == (self.n + 1,)

    def test_compute_invgauss_negloglikel_hessian(self):
        res = compute_invgauss_negloglikel_hessian(
            self.params, self.xn, self.wn, self.eta
        )
        assert res.shape == (self.n + 1, self.n + 1)
