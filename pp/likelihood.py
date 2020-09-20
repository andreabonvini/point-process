from typing import Union

import numpy as np

from pp.distributions import inverse_gaussian


def compute_loglikel(
    eta: np.array, k: float, mus: Union[np.array, float], wn: Union[np.array, float]
):
    logps = np.log(inverse_gaussian(wn, mus, k))
    return np.dot(eta.T, logps)


def likel_invnorm(
    xn: np.array,
    wn: np.array,
    eta: np.array = None,
    thetap0: np.array = None,
    k0: float = None,
    xt: np.array = None,
    wt: float = None,
    max_steps: int = None,
):
    """
    @param xn: xn is a matrix MxN of regressors, each row of xn is associated to the corresponding element of wn.
    @type xn: np.array
    @param wn: wn is a vector Mx1 of observations.
    @type wn: np.array
    @param eta: eta is a vector Mx1 of weights (when missing or empty, then a constant weight is used)
    @type eta: np.array
    @param thetap0:
    thetap0 is a vector Nx1 of coefficients used as starting point for the  newton-rhapson optimization
    (found, e.g., using the uncensored estimation)
    @type thetap0: np.array
    @param k0: k0 is the starting point for the scale parameter (sometimes called lambda).
    @type k0: float
    @param xt: xt is a vector 1xN of regressors, for the censoring part. (IF RIGHT-CENSORING)
    @type xt: np.array
    @param wt: wt is the current value of the future observation. (IF RIGHT-CENSORING)
    @type wt: float
    @param max_steps: max_steps is the maximum number of allowed newton-raphson iterations.
    @type max_steps: int
    @return: [thetap,k,steps,loglikel] (IF NOT RIGHT-CENSORING)
    #TODO What should I return if right-censoring is applied? Check the MATLAB script.
    - thetap:
    is a vector Nx1 of coefficients such that xn*thetap gives the mean of the history-dependent
    inverse gaussian distribution for each observation.
    - k:
    is the scale parameter of the inverse gaussian distribution (sometimes called lambda)
    - steps:
    is the number of newton-raphson iterations used
    - loglikel:
    is the loglikelihood of the observations given the optimized parameters
    @rtype: list
    """
    # Some consistency checks
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

    # TODO CHANGE INITIALIZATION
    if thetap0 is None:
        thetap = np.ones((n, 1))
    else:
        thetap = thetap0
    if k0 is None:
        k = 1.0
    else:
        k = k0
    if eta is None:
        eta = np.ones((m, 1))

    mus = np.dot(xn, thetap)
    # mus.shape : (m,1)
    loglikel = compute_loglikel(eta, k, mus, wn)
    print(loglikel)
    print(wt)
    print(max_steps)
