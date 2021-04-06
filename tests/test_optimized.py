from unittest import TestCase

import numpy as np
from scipy.optimize import approx_fprime

from pp.core.distributions.inverse_gaussian import (
    compute_invgauss_negloglikel,
    compute_invgauss_negloglikel_rc,
)
from pp.model import PointProcessDataset
from pp.optimized.py_regr_likel import ig_gradient, mpfr_ig_gradient_rc

# approx_fprime(xk, f, epsilon, **args)
# f is the function of which to determine the gradient (partial derivatives).
# Should take xk as first argument, other arguments to f can be supplied in *args.
# Should return a scalar, the value of the function at xk.


class TestInverseGaussianDerivatives(TestCase):
    def setUp(self) -> None:
        events = np.array(
            [
                727.391,
                728.297,
                729.188,
                730.062,
                730.984,
                731.93,
                732.875,
                733.828,
                734.781,
                735.711,
            ]
        )
        ar_order = 3
        self.dataset = PointProcessDataset.load(events, p=ar_order)

    def test_ig_gradient(self):
        xk = np.array([1500.0, 0.1, 0.1, 0.1, 0.1])
        k = xk[0]
        thetap = xk[1:].reshape(-1, 1)
        xn = self.dataset.xn
        wn = self.dataset.wn
        eta = self.dataset.eta
        gradient = ig_gradient(xn, thetap, k, wn, eta)
        approximated_gradient = approx_fprime(
            xk, compute_invgauss_negloglikel, 1e-10, xn, wn, eta
        )
        np.testing.assert_array_almost_equal(
            gradient, approximated_gradient, 1, verbose=True
        )

    def test_ig_gradient_rc(self):
        xk = np.array([1500.0, 0.85, 0.4, -0.45, 0.1])
        xt = np.array([1.0, 0.93, 0.953, 0.953]).reshape(-1, 1).T
        rc_eta = 1.0
        thetap = xk[1:].reshape(-1, 1)
        k = xk[0]
        wt = 0.85
        mpfr_gradient = mpfr_ig_gradient_rc(k, thetap, xt, wt, rc_eta)
        approximated_gradient = approx_fprime(
            xk, compute_invgauss_negloglikel_rc, 1e-20, wt, rc_eta, xt
        )
        np.testing.assert_array_almost_equal(
            mpfr_gradient, approximated_gradient, 2, verbose=True
        )
