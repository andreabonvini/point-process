from typing import List, Optional

import numpy as np
from scipy.optimize import LinearConstraint
from scipy.stats import norm

# ------------------------------------------------------------------------------------------------------------
# The following block of code is here just for testing purposes, the real computation for the negloglikelihood
# happens in the c code in pp/optimized/c_regr_likel.c
# ------------------------------------------------------------------------------------------------------------


def igcdf(t: float, mu: float, k: float):  # pragma: no cover
    if t == 0.0:
        return 0.0
    return norm.cdf(np.sqrt(k / t) * (t / mu - 1)) + np.exp(
        (2 * k / mu) + norm.logcdf(-np.sqrt(k / t) * (t / mu + 1))
    )


def compute_invgauss_negloglikel_rc(
    xk: np.ndarray, wt: float, rc_eta: float, xt: np.ndarray
) -> float:
    # Retrieve useful variable
    k, thetap = xk[0], xk[1:]
    thetap = thetap.reshape(-1, 1)
    rc_mu = np.dot(xt, thetap)[0, 0]
    return -rc_eta * np.log(1.0 - igcdf(wt, rc_mu, k))


def compute_invgauss_negloglikel(
    xk: np.ndarray, xn: np.ndarray, wn: np.ndarray, eta: np.ndarray,
) -> float:
    # Retrieve useful variables
    k, thetap = xk[0], xk[1:]
    thetap = thetap.reshape(-1, 1)
    mus = np.dot(xn, thetap)
    # Compute negative log-likelihood
    arg = k / (2 * np.pi * wn ** 3)
    logps = np.log(np.sqrt(arg)) - (k * (wn - mus) ** 2) / (2 * mus ** 2 * wn)
    return -np.dot(eta.T, logps)[0, 0]


# ------------------------------------------------------------------------------------------------------------


class InverseGaussianConstraints:  # pragma: no cover
    # FIXME This constraints are't actually forced by now. (nlopt NEWTON method doesn't support linear constraints)
    #  They're here just as a reminder on how that could be done..
    def __init__(self, xn: np.ndarray):
        """
        Args:
            xn: (n,p+1) or (n,p) matrix where
            n=number_of_samples.
            p=AR order.
            +1 is to account for the bias parameter in case it is used by the model.
        """
        self._samples = xn

    def __call__(self) -> List[LinearConstraint]:
        return self._compute_constraints()

    def _compute_constraints(self) -> List[LinearConstraint]:
        big_n = 1e6
        # Let's firstly define the positivity constraint for the parameter lambda
        n_samples, n_var = self._samples.shape
        # We also have to take into account the scale parameter
        n_var += 1
        A_lambda = np.identity(n_var)
        lb_lambda = np.ones((n_var,)) * (-big_n)
        # lambda (the scale parameter) is the only parameter that must be strictly positive
        lb_lambda[0] = 1e-7
        ub_lambda = np.ones((n_var,)) * big_n

        # The theta parameters can take both positive and negative values, however the mean estimate from the AR model
        # should always be positive.
        # We stack a vector of zeros of shape (n_samples,1) as the first column of A, this way the constraints
        # definition will not interfer with the choose of the lambda parameter (aka scale parameter, aka k).
        A_theta = np.hstack([np.zeros((n_samples, 1)), self._samples])
        lb_theta = np.zeros((n_samples,))
        ub_theta = np.ones((n_samples,)) * big_n
        return [
            LinearConstraint(A_lambda, lb_lambda, ub_lambda),
            LinearConstraint(A_theta, lb_theta, ub_theta),
        ]


def likel_invgauss_consistency_check(
    xn: np.ndarray,
    wn: np.ndarray,
    xt: Optional[np.ndarray] = None,
    thetap0: Optional[np.ndarray] = None,
):
    m, n = xn.shape
    if wn.shape != (m, 1):
        raise ValueError(
            f"Since xn has shape {xn.shape}, wn should be of shape ({m},1).\n"
            f"Instead wn has shape {wn.shape}"
        )
    if xt is not None and xt.shape != (1, n):
        raise ValueError(
            f"Since xn has shape {xn.shape}, xt should be of shape (1,{n}).\n"
            f"Instead xt has shape {xt.shape}"
        )
    if thetap0 is not None and thetap0.shape != (n, 1):
        raise ValueError(
            f"Since xn has shape {xn.shape}, thetap0 should be of shape ({n},1).\n"
            f"Instead thetap0 has shape {thetap0.shape}"
        )


def _compute_mus(thetap: np.ndarray, xn: np.ndarray) -> np.ndarray:  # pragma: no cover
    return np.dot(xn, thetap)
