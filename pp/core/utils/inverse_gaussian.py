from typing import Union

import numpy as np

from pp.core.model import InterEventDistribution, PointProcessModel, PointProcessResult


def build_ig_model(thetap: np.ndarray, k: float, hasTheta0: bool) -> PointProcessModel:
    expected_shape = (thetap.shape[0] - 1,) if hasTheta0 else (thetap.shape[0],)

    def ig_model(inter_event_times: np.ndarray) -> PointProcessResult:
        if inter_event_times.shape != expected_shape:
            raise ValueError(
                f"The inter-event times shape ({inter_event_times.shape})"
                f" is incompatible with the inter-event times shape used for training ({expected_shape})"
            )
        # reverse order
        inter_event_times = inter_event_times[::-1]
        # append 1 if hasTheta0
        inter_event_times = (
            np.concatenate(([1], inter_event_times)) if hasTheta0 else inter_event_times
        )
        # reshape from (n,) to (n,1)
        inter_event_times = inter_event_times.reshape(-1, 1)
        mu = np.dot(thetap, inter_event_times)[0]
        sigma = np.sqrt(mu ** 3 / k)
        return PointProcessResult(mu, sigma)

    return PointProcessModel(
        model=ig_model,
        expected_shape=expected_shape,
        distribution=InterEventDistribution.INVERSE_GAUSSIAN,
        ar_order=expected_shape[0],
        hasTheta0=hasTheta0,
    )


def inverse_gaussian(
    xs: Union[np.array, float], mus: Union[np.array, float], lamb: float
) -> np.ndarray:
    """
    @param xs: points or point in which evaluate the probabilty
    @type xs: np.array or float
    @param mus: inverse gaussian means or mean
    @type mus: np.array or float
    @param lamb: inverse gaussian scaling factor
    @type lamb: float
    @return: p: probability values, 0 < p < 1
    @rtype: np.array
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
    return np.sqrt(arg) * np.exp((-lamb * (xs - mus) ** 2) / (2 * mus ** 2 * xs))


def unpack_invgauss_params(params: np.array, m: int, n: int):
    return params[0], params[1 : 1 + m].reshape((m, 1)), params[1 + m :].reshape((n, 1))


def likel_invgauss_consistency_check(
    xn: np.array,
    wn: np.array,
    eta: Union[np.array, None],
    xt: Union[np.array, None],
    thetap0: Union[np.array, None],
):
    m, n = xn.shape
    if wn.shape != (m, 1):
        raise ValueError(
            f"Since xn has shape {xn.shape}, wn should be of shape ({m},1).\n"
            f"Instead wn has shape {wn.shape}"
        )
    if eta is not None and eta.shape != (m, 1):
        raise ValueError(
            f"Since xn has shape {xn.shape}, eta should be of shape ({m},1).\n"
            f"Instead eta has shape {eta.shape}"
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


def compute_invgauss_negloglikel(params: np.array, xn: np.array, wn: np.array) -> float:
    """
    ALERT: Remember that the parameters that we want to optimize are just k, eta and thetap
    TODO Handle the case in which eta is constant
    """

    m, n = xn.shape
    k_param, eta_params, thetap_params = unpack_invgauss_params(params, m, n)

    mus = np.dot(xn, thetap_params).reshape((m, 1))
    logps = np.log(inverse_gaussian(wn, mus, k_param))
    return -np.dot(eta_params.T, logps)[0, 0]


def compute_invgauss_negloglikel_grad(
    params: np.array, xn: np.array, wn: np.array
) -> np.ndarray:
    """
    returns the vector of the first-derivatives of the negloglikelihood w.r.t to each parameter
    """

    m, n = xn.shape
    # Retrieve the useful variables
    k_param, eta_params, thetap_params = unpack_invgauss_params(params, m, n)
    mus = np.dot(xn, thetap_params).reshape((m, 1))

    # Compute the gradient for k
    tmp = -1 / k_param + (wn - mus) ** 2 / (mus ** 2 * wn)
    k_grad = np.dot((eta_params / 2).T, tmp)

    # Compute the gradient for eta[0]...eta[m-1]
    eta_grad = -1 * np.log(inverse_gaussian(wn, mus, k_param))

    # Compute the gradient form thetap[0]...thetap[n-1]
    tmp = -1 * k_param * eta_params * (wn - mus) / mus ** 3
    thetap_grad = np.dot(tmp.T, xn).T

    # Return all the gradients as a single vector of shape (n+m+1,)
    return np.vstack([k_grad, eta_grad, thetap_grad]).squeeze(1)


def compute_invgauss_negloglikel_hessian(
    params: np.array, xn: np.array, wn: np.array
) -> np.ndarray:
    """
    returns the vector of the second-derivatives of the negloglikelihood w.r.t to each
    parameter
    """

    m, n = xn.shape
    # Retrieve the useful variables
    k_param, eta_params, thetap_params = unpack_invgauss_params(params, m, n)
    mus = np.dot(xn, thetap_params).reshape((m, 1))

    # Initialize hessian matrix
    hess = np.zeros((1 + m + n, 1 + m + n))

    # We populate the hessian as a upper triangular matrix
    # by filling the rows starting from the main diagonal

    # Partial derivatives w.r.t. k
    kk = np.sum(eta_params) * 1 / (2 * k_param ** 2)
    keta = 1 / 2 * (-1 / k_param + (wn - mus) ** 2 / (mus ** 2 * wn))
    tmp = eta_params * (wn - mus) / mus ** 3
    ktheta = -np.dot(tmp.T, xn).T

    hess[0, 0] = kk
    hess[0, 1 : (1 + m)] = keta.squeeze(1)
    hess[0, (1 + m) : (1 + m + n)] = ktheta.squeeze(1)

    # All the partial derivatives in the form eta_j\eta_q are null
    for i in range(1, m + 1):
        for j in range(i, m + 1):
            hess[i, j] = 0

    # TODO is there a smarter way? (eta_j\theta_q)
    for i in range(1, m + 1):
        for j in range(m + 1 + i, m + 1 + n):
            hess[i, j] = (
                -k_param
                * (xn[i - 1, j - m - 1])
                * (wn[i - 1] - mus[i - 1])
                / mus[i - 1] ** 3
            )

    # TODO is there a smarter way? (theta_j\theta_q)
    for i in range(m + 1, m + n + 1):
        for j in range(m + 1 + i, m + 1 + n):
            tmp1 = xn[:, i - m - 1] * xn[:, j - m - 1]
            tmp2 = eta_params * k_param * (3 * wn - 2 * mus) / (mus ** 4)
            hess[i, j] = np.dot(tmp1.T, tmp2)

    # Populate the rest of the matrix
    hess = np.where(hess, hess, hess.T)
    return hess
