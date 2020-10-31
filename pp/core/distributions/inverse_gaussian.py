from abc import ABC
from typing import List, Union

import numpy as np
from scipy.optimize import LinearConstraint

from pp.core.model import (
    InterEventDistribution,
    PointProcessConstraint,
    PointProcessModel,
    PointProcessResult,
)


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
        maximum_event_interval = 2000
        # Let's firstly define the positivity contraint for the parameter lambda
        n_samples, n_var = self._samples.shape
        # We also have to take into account the scale parameter
        n_var += 1
        A_lambda = np.identity(n_var)
        lb_lambda = np.ones((n_var,)) * -np.inf
        # lambda (the scale parameter) is the only parameter that must be strictly positive
        lb_lambda[0] = 1e-7
        ub_lambda = np.ones((n_var,)) * np.inf

        # The theta parameters can take both positive and negative values, however the mean estimate from the AR model
        # should always be positive.
        # We stack a vector of zeros of shape (n_samples,1) as the first column of A, this way the constraints
        # definition will not interfer with the choose of the lambda parameter (aka scale parameter).
        A_theta = np.hstack([np.zeros((n_samples, 1)), self._samples])
        lb_theta = np.zeros((n_samples,))
        ub_theta = np.ones((n_samples,)) * maximum_event_interval
        return [
            LinearConstraint(A_lambda, lb_lambda, ub_lambda),
            LinearConstraint(A_theta, lb_theta, ub_theta),
        ]


def build_ig_model(
    theta: np.ndarray,
    k: float,
    hasTheta0: bool,
    results: List[float],
    params_history: List[np.ndarray],
) -> PointProcessModel:
    expected_shape = (theta.shape[0] - 1,) if hasTheta0 else (theta.shape[0],)

    def ig_model(inter_event_times: np.ndarray) -> PointProcessResult:
        if inter_event_times.shape != expected_shape:
            raise ValueError(
                f"The inter-event times shape ({inter_event_times.shape})"
                f" is incompatible with the inter-event times shape used for training ({expected_shape})"
            )
        # append 1 if hasTheta0
        inter_event_times = (
            np.concatenate(([1], inter_event_times)) if hasTheta0 else inter_event_times
        )
        # reshape from (n,) to (n,1)
        inter_event_times = inter_event_times.reshape(-1, 1)
        mu = np.dot(theta, inter_event_times)[0]
        sigma = np.sqrt(mu ** 3 / k)
        return PointProcessResult(mu, sigma)

    return PointProcessModel(
        model=ig_model,
        expected_shape=expected_shape,
        theta=theta,
        k=k,
        results=results,
        params_history=params_history,
        distribution=InterEventDistribution.INVERSE_GAUSSIAN,
        ar_order=expected_shape[0],
        hasTheta0=hasTheta0,
    )


def inverse_gaussian(
    xs: Union[np.array, float], mus: Union[np.array, float], lamb: float
) -> Union[np.ndarray, float]:
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

    elif isinstance(xs, np.ndarray) or isinstance(mus, np.ndarray):
        raise TypeError(
            f"xs: {type(xs)}\n"
            f"mus: {type(mus)}\n"
            f"xs and mus should be either both np.array or both float"
        )
    arg = lamb / (2 * np.pi * xs ** 3)
    ps = np.sqrt(arg) * np.exp((-lamb * (xs - mus) ** 2) / (2 * mus ** 2 * xs))
    return ps


def likel_invgauss_consistency_check(
    xn: np.array,
    wn: np.array,
    xt: Union[np.array, None],
    thetap0: Union[np.array, None],
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


def _compute_mus(thetap: np.array, xn: np.ndarray):
    return np.dot(xn, thetap)


def _compute_k_grad(eta, k, wn, mus) -> float:
    return 1 / 2 * np.dot(eta.T, -1 / k + (wn - mus) ** 2 / (mus ** 2 * wn))[0, 0]


def _compute_theta_grad(xn, eta, k, wn, mus) -> np.ndarray:
    tmp = -1 * k * eta * (wn - mus) / mus ** 3
    return np.dot(xn.T, tmp)


def compute_invgauss_negloglikel(
    params: np.array, xn: np.array, wn: np.array, eta: np.array
) -> float:
    """
    ALERT: Remember that the parameters that we want to optimize are just k, and thetap
    """
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
    ps = inverse_gaussian(wn, mus, k_param)
    # 1. is just to avoid log(0) in case the ps are really small
    logps = np.log(1.0 + ps)
    return -np.dot(eta.T, logps)[0, 0]


def compute_invgauss_negloglikel_grad(
    params: np.array, xn: np.array, wn: np.array, eta: np.array
) -> np.ndarray:
    """
    returns the vector of the first-derivatives of the negloglikelihood w.r.t to each parameter
    """

    n, m = xn.shape
    # Retrieve the useful variables
    k, theta = params[0], params[1:].reshape((len(params) - 1, 1))
    mus = _compute_mus(theta, xn)
    k_grad = _compute_k_grad(eta, k, wn, mus)
    theta_grad = _compute_theta_grad(xn, eta, k, wn, mus)

    # Return all the gradients as a single vector of shape (p+1+1,) or (p+1,) if theta0 is not accounted.
    return np.vstack([[[k_grad]], theta_grad]).squeeze(1)


def compute_invgauss_negloglikel_hessian(
    params: np.array, xn: np.array, wn: np.array, eta: np.array
) -> np.ndarray:
    """
    returns the vector of the second-derivatives of the negloglikelihood w.r.t to each
    parameter
    """

    n, m = xn.shape
    # Retrieve the useful variables
    k, theta = params[0], params[1:].reshape((m, 1))
    mus = np.dot(xn, theta).reshape((n, 1))
    kk = np.sum(eta / 2) * 1 / (k ** 2)
    tmp = -eta * (wn - mus) / mus ** 3

    ktheta = np.dot(tmp.T, xn)
    tmp = k * eta * ((3 * wn - 2 * mus) / mus ** 4)
    thetatheta = np.dot(xn.T, xn * tmp)
    m, _ = thetatheta.shape
    k_theta_hess = np.zeros((m + 1, m + 1))
    k_theta_hess[1:, 1:] = thetatheta
    k_theta_hess[0, 0] = kk
    k_theta_hess[0, 1:] = k_theta_hess[1:, 0] = ktheta.squeeze()
    return k_theta_hess
