import numpy as np

from pp.distributions import inverse_gaussian


def compute_loglikel(eta: np.array, k: float, mus: np.array, wn: np.array):
    logps = np.log(inverse_gaussian(wn, mus, k))
    return np.dot(eta.T, logps)[0, 0]


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

    pass
