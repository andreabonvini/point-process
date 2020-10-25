from unittest import TestCase

import numpy as np

from pp.core.model import InterEventDistribution
from pp.core.utils.inverse_gaussian import (
    build_ig_model,
    compute_invgauss_negloglikel,
    compute_invgauss_negloglikel_grad,
    compute_invgauss_negloglikel_hessian,
    inverse_gaussian,
    likel_invgauss_consistency_check,
    unpack_invgauss_params,
)


class TestInverseGaussian(TestCase):
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
        self.wn = np.ones((self.m, 1))
        self.params = np.vstack(
            [0.5, np.ones((self.m, 1)), np.ones((self.n, 1))]
        ).squeeze(1)
        self.eta = np.ones((self.m, 1))
        self.thetap = np.ones((self.n, 1))

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

    def test_compute_invgauss_negloglikel(self):
        res = compute_invgauss_negloglikel(self.params, self.xn, self.wn)
        assert type(res) == np.float64

    def test_compute_invgauss_negloglikel_grad(self):
        res = compute_invgauss_negloglikel_grad(self.params, self.xn, self.wn)
        assert res.shape == (self.n + self.m + 1,)

    def test_compute_invgauss_negloglikel_hessian(self):
        res = compute_invgauss_negloglikel_hessian(self.params, self.xn, self.wn)
        assert res.shape == (self.n + self.m + 1, self.n + self.m + 1)

    def test_unpack_invgauss_params(self):
        res = unpack_invgauss_params(self.params, self.m, self.n)
        assert type(res[0]) == np.float64
        assert res[1].shape == (self.m, 1)
        assert res[2].shape == (self.n, 1)

    def test_likel_invgauss_consistency_check_good(self):
        res = likel_invgauss_consistency_check(
            xn=self.xn, wn=self.wn, eta=self.eta, xt=None, thetap0=self.thetap
        )
        self.assertIsNone(res)

    def test_likel_invgauss_consistency_check_wn_bad(self):
        with self.assertRaises(ValueError):
            likel_invgauss_consistency_check(
                xn=self.xn,
                wn=np.ones((self.m - 1, 1)),
                eta=self.eta,
                xt=None,
                thetap0=self.thetap,
            )

    def test_likel_invgauss_consistency_check_xt_bad(self):
        with self.assertRaises(ValueError):
            likel_invgauss_consistency_check(
                xn=self.xn,
                wn=self.wn,
                eta=self.eta,
                xt=np.ones((1, self.n - 1)),
                thetap0=self.thetap,
            )

    def test_likel_invgauss_consistency_check_eta_bad(self):
        with self.assertRaises(ValueError):
            likel_invgauss_consistency_check(
                xn=self.xn,
                wn=self.wn,
                eta=np.ones((self.m - 1, 1)),
                xt=None,
                thetap0=self.thetap,
            )

    def test_likel_invgauss_consistency_check_thetap_bad(self):
        with self.assertRaises(ValueError):
            likel_invgauss_consistency_check(
                xn=self.xn,
                wn=self.wn,
                eta=self.eta,
                xt=None,
                thetap0=np.ones((self.n - 1, 1)),
            )

    def test_build_ig_model_hasTheta0_true(self):
        hasTheta0 = True
        ar_order = 3
        thetap = np.ones(ar_order + 1) / (ar_order + 1)
        k = 1.00
        pp_model = build_ig_model(thetap, k, hasTheta0)
        inter_event_times = np.ones((ar_order,))
        res = pp_model(inter_event_times)
        self.assertEqual(res.mu, 1.00)
        self.assertEqual(res.sigma, 1.00)
        # FIXME move to test_model the code below
        self.assertEqual(pp_model.hasTheta0, hasTheta0)
        self.assertEqual(pp_model.ar_order, ar_order)
        self.assertEqual(pp_model.distribution, InterEventDistribution.INVERSE_GAUSSIAN)
        self.assertEqual(pp_model.expected_input_shape, (ar_order,))

    def test_ig_model_wrong_usage(self):
        hasTheta0 = True
        ar_order = 3
        thetap = np.ones(ar_order + 1) / (ar_order + 1)
        k = 1.00
        pp_model = build_ig_model(thetap, k, hasTheta0)
        wrong_shape_inter_event_times = np.ones((ar_order + 1,))
        with self.assertRaises(ValueError):
            pp_model(wrong_shape_inter_event_times)
