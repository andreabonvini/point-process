from typing import Optional

import numpy as np
from scipy.optimize import minimize

from pp.core.model import InterEventDistribution, PointProcessModel
from pp.core.utils.constraints import greater_than_zero
from pp.core.utils.inverse_gaussian import (
    build_ig_model,
    compute_invgauss_negloglikel,
    compute_invgauss_negloglikel_grad,
    compute_invgauss_negloglikel_hessian,
    likel_invgauss_consistency_check,
    unpack_invgauss_params,
)


def inverse_gaussian_maximizer(
    xn: np.array,
    wn: np.array,
    max_steps: int = 1000,
    eta0: Optional[np.array] = None,
    thetap0: Optional[np.array] = None,
    k0: Optional[float] = None,
    xt: Optional[np.array] = None,
    wt: Optional[float] = None,
) -> PointProcessModel:
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
        thetap0 = np.ones((n, 1)) / n
    if k0 is None:
        k0 = 0.5
    if eta0 is None:
        eta0 = np.ones((m, 1)) / m

    # In order to optimize the parameters with scipy.optimize.minimize we need to pack all of our parameters in a
    # vector of shape (1+m+n,)
    params0 = np.vstack([k0, eta0, thetap0]).squeeze(1)

    # we want our parameters to be greater than 0
    cons = greater_than_zero(m + n + 1)

    optimization_result = minimize(
        fun=compute_invgauss_negloglikel,
        x0=params0,
        jac=compute_invgauss_negloglikel_grad,
        hess=compute_invgauss_negloglikel_hessian,
        method="trust-constr",
        args=(xn, wn),
        constraints=cons,
        options={"maxiter": max_steps, "disp": True},
    )
    optimal_parameters = optimization_result.x
    k_param, eta_params, thetap_params = unpack_invgauss_params(
        optimal_parameters, m, n
    )
    model = build_ig_model(k_param, eta_params, thetap_params)
    return PointProcessModel(
        model=model, distribution=InterEventDistribution.INVERSE_GAUSSIAN, ar_order=n
    )
