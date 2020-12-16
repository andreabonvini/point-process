from abc import ABC
from typing import List, Optional

import numpy as np
from scipy.misc import derivative
from scipy.optimize import LinearConstraint
from scipy.stats import norm

from pp.model import PointProcessConstraint


class InverseGaussianConstraints(PointProcessConstraint, ABC):
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
        # Let's firstly define the positivity contraint for the parameter lambda
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
        # definition will not interfer with the choose of the lambda parameter (aka scale parameter).
        A_theta = np.hstack([np.zeros((n_samples, 1)), self._samples])
        lb_theta = np.zeros((n_samples,))
        ub_theta = np.ones((n_samples,)) * big_n
        return [
            LinearConstraint(A_lambda, lb_lambda, ub_lambda),
            LinearConstraint(A_theta, lb_theta, ub_theta),
        ]


def igcdf(t: float, mu: float, k: float):  # pragma: no cover
    return norm.cdf(np.sqrt(k / t) * (t / mu - 1)) + np.exp(
        (2 * k / mu) + norm.logcdf(-np.sqrt(k / t) * (t / mu + 1))
    )


def igpdf(t: float, mu: float, k: float):  # pragma: no cover
    arg = k / (2 * np.pi * t ** 3)
    return np.nan_to_num(
        np.sqrt(arg) * np.exp((-k * (t - mu) ** 2) / (2 * mu ** 2 * t))
    )


def _log_inverse_gaussian(xs: np.array, mus: np.array, lamb: float) -> np.ndarray:
    """
    Args:
        xs: points or point in which evaluate the probabilty
        mus: inverse gaussian means or mean
        lamb: inverse gaussian scaling factor

    Returns:
        p: probability values (or value), 0 < p < 1
    """

    if isinstance(xs, np.ndarray) and isinstance(mus, np.ndarray):
        if xs.shape != mus.shape:
            raise ValueError(
                f"{xs.shape}!={mus.shape}.\n"
                "xs and mus should have the same shape if they're both np.array"
            )

    elif not isinstance(xs, np.ndarray) or not isinstance(mus, np.ndarray):
        raise TypeError(
            f"xs: {type(xs)}\n"
            f"mus: {type(mus)}\n"
            f"xs and mus should be both np.array"
        )
    arg = lamb / (2 * np.pi * xs ** 3)
    ps = np.sqrt(arg) * np.exp((-lamb * (xs - mus) ** 2) / (2 * mus ** 2 * xs))
    return np.log(ps)


def likel_invgauss_consistency_check(
    xn: np.array,
    wn: np.array,
    xt: Optional[np.array] = None,
    thetap0: Optional[np.array] = None,
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


def _compute_mus(thetap: np.array, xn: np.ndarray) -> np.array:
    return np.dot(xn, thetap)


def _compute_k_grad(eta, k, wn, mus) -> float:
    return 1 / 2 * np.dot(eta.T, -1 / k + (wn - mus) ** 2 / (mus ** 2 * wn))[0, 0]


def _compute_theta_grad(xn, eta, k, wn, mus) -> np.ndarray:
    tmp = -1 * k * eta * (wn - mus) / mus ** 3
    return np.dot(xn.T, tmp)


def compute_lambda(mu: float, k: float, time: float):
    return igpdf(time, mu, k) / (1 - igcdf(time, mu, k))


def _compute_invgauss_negloglikel_norc(
    params: np.array, xn: np.array, wn: np.array, eta: np.array,
) -> float:
    n, m = xn.shape
    k_param, theta_params = params[0], params[1:]
    # if k_param < 0:
    #    raise Exception(ValueError(f"Illegal value for lambda!\nk:{params[0]} < 0 "))
    mus = np.dot(xn, theta_params).reshape((n, 1))

    # if any(mu <= 0 for mu in mus):
    #     raise Exception(ValueError(f"Illegal value for theta!\nSome predictions "
    #                               f"(i.e. the dot product between xn and theta) are < 0\n"
    #                               f"xn:{xn}\n"
    #                               f"theta:{params[1:]}\n"
    #                               f"predictions:{mus}"))
    logps = _log_inverse_gaussian(wn, mus, k_param)
    return -np.dot(eta.T, logps)[0, 0]


def compute_invgauss_negloglikel(
    params: np.array,
    xn: np.array,
    wn: np.array,
    eta: np.array,
    xt: Optional[np.array] = None,
    wt: Optional[float] = None,
) -> float:
    k_param, theta_params = params[0], params[1:]
    # If right censoring is applied...
    if wt is not None and xt is not None:  # pragma: no cover
        rc_mu = np.dot(xt, theta_params)[0]
        rc_eta = eta[0][-1]
        # FIXME is 1e-14 safe?
        return _compute_invgauss_negloglikel_norc(
            params, xn, wn, eta
        ) - rc_eta * np.log(1 - igcdf(wt, rc_mu, k_param) + 1e-14)
    else:
        return _compute_invgauss_negloglikel_norc(params, xn, wn, eta)


def compute_invgauss_negloglikel_grad(
    params: np.array,
    xn: np.array,
    wn: np.array,
    eta: np.array,
    xt: Optional[np.array] = None,
    wt: Optional[float] = None,
) -> np.ndarray:
    """
    returns the vector of the first-derivatives of the negloglikelihood w.r.t to each parameter
    """

    # Retrieve the useful variables
    k, theta = params[0], params[1:].reshape((len(params) - 1, 1))
    mus = _compute_mus(theta, xn)
    k_grad = _compute_k_grad(eta, k, wn, mus)
    theta_grad = _compute_theta_grad(xn, eta, k, wn, mus)
    # If right censoring is applied...
    if wt is not None and xt is not None:  # pragma: no cover
        rc_eta = eta[0][-1]
        rc_mu = np.dot(xt, theta)[0, 0]
        rc_dk = np.nan_to_num(_right_censoring_derivative_k(k, wt, rc_mu, rc_eta))
        rc_dtheta = (
            np.nan_to_num(_right_censoring_derivative_mu(k, wt, rc_mu, rc_eta)) * xt.T
        )
        k_grad += rc_dk
        theta_grad += rc_dtheta
    # Return all the gradients as a single vector of shape (p+1+1,) or (p+1,) if theta0 is not accounted.
    return np.vstack([[[k_grad]], theta_grad]).squeeze(1)


def _compute_invgauss_negloglikel_hessian_values_norc(
    k: float, theta: np.array, xn: np.array, wn: np.array, eta: np.array
):
    n, _ = xn.shape
    # Retrieve the useful variables
    mus = np.dot(xn, theta).reshape((n, 1))
    kk = np.sum(eta / 2) * 1 / (k ** 2)
    tmp = -eta * (wn - mus) / mus ** 3
    ktheta = np.dot(tmp.T, xn)
    tmp = k * eta * ((3 * wn - 2 * mus) / mus ** 4)
    thetatheta = np.dot(xn.T, xn * tmp)
    return kk, ktheta, thetatheta


def compute_invgauss_negloglikel_hessian(
    params: np.array,
    xn: np.array,
    wn: np.array,
    eta: np.array,
    xt: Optional[np.array] = None,
    wt: Optional[float] = None,
) -> np.ndarray:
    """
    returns the vector of the second-derivatives of the negloglikelihood w.r.t to each
    parameter
    """
    _, m = xn.shape
    k, theta = params[0], params[1:].reshape((m, 1))
    kk, ktheta, thetatheta = _compute_invgauss_negloglikel_hessian_values_norc(
        k, theta, xn, wn, eta
    )
    # If right censoring is applied...
    if wt is not None and xt is not None:  # pragma: no cover
        rc_mu = np.dot(xt, theta)[0, 0]
        rc_eta = eta[0][-1]
        rc_dkk = np.nan_to_num(_right_censoring_derivative_kk(wt, rc_mu, k, rc_eta))
        rc_dthetatheta = np.nan_to_num(
            _right_censoring_derivative_mumu(wt, rc_mu, k, rc_eta)
        ) * np.dot(xt.T, xt)
        rc_dktheta = (
            np.nan_to_num(_right_censoring_derivative_kmu(wt, rc_mu, k, rc_eta)) * xt
        )
        thetatheta += rc_dthetatheta
        kk += rc_dkk
        ktheta += rc_dktheta

    m, _ = thetatheta.shape
    k_theta_hess = np.zeros((m + 1, m + 1))
    k_theta_hess[1:, 1:] = thetatheta
    k_theta_hess[0, 0] = kk
    k_theta_hess[0, 1:] = k_theta_hess[1:, 0] = ktheta.squeeze()

    return k_theta_hess


def _right_censoring_derivative_k(t, mu, k, etan):  # pragma: no cover
    # FIXME is 1e-13 safe?
    mul1 = etan / (1e-13 + 1 - igcdf(t, mu, k))
    add1 = norm.pdf(np.sqrt(k / t) * (t / mu - 1)) * (
        1 / (2 * t * np.sqrt(k / t)) * (t / mu - 1)
    )
    add2 = 2 / mu * np.exp(2 * k / mu + norm.logcdf(-np.sqrt(k / t) * (t / mu + 1)))
    add3 = np.exp(2 * k / mu + norm.logpdf(-np.sqrt(k / t) * (t / mu + 1))) * (
        -1 / (2 * t * np.sqrt(k / t)) * (t / mu + 1)
    )
    return mul1 * (add1 + add2 + add3)


def _right_censoring_derivative_theta(t, mu, k, etan, xt):  # pragma: no cover
    # FIXME is 1e-13 safe?
    mul1 = etan / (1e-13 + 1 - igcdf(t, mu, k))
    add1 = norm.pdf(np.sqrt(k / t) * (t / mu - 1)) * (-np.sqrt(k / t) * t / mu ** 2)
    add2 = (-2 * k / mu ** 2) * np.exp(
        2 * k / mu + norm.logcdf(-np.sqrt(k / t) * (t / mu + 1))
    )
    add3 = np.exp(2 * k / mu + norm.logpdf(-np.sqrt(k / t) * (t / mu + 1))) * (
        np.sqrt(k / t) * t / mu ** 2
    )
    return mul1 * (add1 + add2 + add3) * xt.T


def _right_censoring_derivative_mu(t, mu, k, etan):  # pragma: no cover
    # FIXME is 1e-13 safe?
    mul1 = etan / (1e-13 + 1 - igcdf(t, mu, k))
    add1 = norm.pdf(np.sqrt(k / t) * (t / mu - 1)) * (-np.sqrt(k / t) * t / mu ** 2)
    add2 = (-2 * k / mu ** 2) * np.exp(
        2 * k / mu + norm.logcdf(-np.sqrt(k / t) * (t / mu + 1))
    )
    add3 = np.exp(2 * k / mu + norm.logpdf(-np.sqrt(k / t) * (t / mu + 1))) * (
        np.sqrt(k / t) * t / mu ** 2
    )
    return mul1 * (add1 + add2 + add3)


def _right_censoring_derivative_kk(wt, rc_mu, k, rc_eta):  # pragma: no cover
    def _dk(x: float):
        return _right_censoring_derivative_k(wt, rc_mu, x, etan=rc_eta)

    return derivative(func=_dk, x0=k, dx=0.5)  # TODO that dx=0.5 may be way too high


def _right_censoring_derivative_mumu(wt, rc_mu, k, rc_eta):  # pragma: no cover
    def _dmu(x: float):
        return _right_censoring_derivative_mu(wt, x, k, etan=rc_eta)

    return derivative(func=_dmu, x0=rc_mu, dx=1e-3)


def _right_censoring_derivative_kmu(wt, rc_mu, k, rc_eta):  # pragma: no cover
    def _dkwrtmu(x: float):
        return _right_censoring_derivative_k(wt, x, k, rc_eta)

    return derivative(func=_dkwrtmu, x0=rc_mu, dx=1e-4)
