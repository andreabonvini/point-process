import numpy as np
from scipy.optimize import minimize

from pp.distributions import inverse_gaussian
from pp.utils import likel_invgauss_consistency_check, unpack_invgauss_params


def compute_invgauss_negloglikel(params: np.array, xn: np.array, wn: np.array):
    """
    ALERT: Remember that the parameters that we want to optimize are just k, eta and thetap
    #TODO Handle the case in which eta is constant
    """

    m, n = xn.shape
    k_param, eta_params, thetap_params = unpack_invgauss_params(params, m, n)

    mus = np.dot(xn, thetap_params).reshape((m, 1))
    logps = np.log(inverse_gaussian(wn, mus, k_param))
    return -np.dot(eta_params.T, logps)[0, 0]


def compute_invgauss_negloglikel_grad(params: np.array, xn: np.array, wn: np.array):
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


def compute_invgauss_negloglikel_hessian(params: np.array, xn: np.array, wn: np.array):
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


def likel_invnorm(
    xn: np.array,
    wn: np.array,
    eta0: np.array = None,
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
    @param eta0: eta is a vector Mx1 of weights (when missing or empty, then a constant weight of 1s is used)
    @type eta0: np.array
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
    """

    # Some consistency checks
    likel_invgauss_consistency_check(xn, wn, eta0, xt, thetap0)

    # TODO CHANGE INITIALIZATION
    m, n = xn.shape
    if thetap0 is None:
        thetap0 = np.ones((n, 1))
    if k0 is None:
        k0 = 0.5
    if eta0 is None:
        eta0 = np.ones((m, 1))

    # In order to optimize the parameters with scipy.optimize.minimize we need to pack all of our parameters in a
    # vector of shape (1+m+n,)
    params0 = np.vstack([k0, eta0, thetap0]).squeeze(1)

    return minimize(
        fun=compute_invgauss_negloglikel,
        x0=params0,
        jac=compute_invgauss_negloglikel_grad,
        args=(xn, wn),
    )
