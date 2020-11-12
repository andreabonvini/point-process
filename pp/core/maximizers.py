from abc import ABC
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from pp.core.distributions.inverse_gaussian import (
    InverseGaussianConstraints,
    build_ig_model,
    compute_invgauss_negloglikel,
    compute_invgauss_negloglikel_grad,
    compute_invgauss_negloglikel_hessian,
    likel_invgauss_consistency_check,
)
from pp.model import PointProcessDataset, PointProcessMaximizer, PointProcessModel


class InverseGaussianMaximizer(PointProcessMaximizer, ABC):
    def __init__(
        self,
        dataset: PointProcessDataset,
        max_steps: int = 1000,
        theta0: Optional[np.array] = None,
        k0: Optional[float] = None,
    ):
        """
            Args:
                dataset: PointProcessDataset to use for the regression.
                max_steps: max_steps is the maximum number of allowed iterations of the optimization process.
                theta0: is a vector of shape (p,1) (or (p+1,1) if teh dataset was created with the hasTheta0 option)
                 of coefficients used as starting point for the optimization process.
                k0: is the starting point for the scale parameter (sometimes called lambda).
            Returns:
                PointProcessModel
            """
        self.dataset = dataset
        self.max_steps = max_steps
        self.theta0 = theta0
        self.k0 = k0
        self.n, self.m = self.dataset.xn.shape
        # Some consistency checks
        likel_invgauss_consistency_check(
            self.dataset.xn, self.dataset.wn, self.dataset.xt, self.theta0
        )

    def train(self) -> PointProcessModel:

        params_history = []
        results = []

        def _save_history(params: np.array, state):  # pragma: no cover
            results.append(
                compute_invgauss_negloglikel(
                    params, self.dataset.xn, self.dataset.wn, self.dataset.eta
                )
            )
            params_history.append(params)

        # TODO change initialization
        if self.theta0 is None:
            self.theta0 = np.ones((self.m, 1)) / self.m
        if self.k0 is None:
            self.k0 = 1200

        # In order to optimize the parameters with scipy.optimize.minimize we need to pack all of our parameters in a
        # vector of shape (1+p,) or (1+1+p,) if hasTheta0
        params0 = np.vstack((self.k0, self.theta0)).squeeze(1)

        cons = InverseGaussianConstraints(self.dataset.xn)()
        # it's ok to have cons as a list of LinearConstrainsts if we're using the "trust-constr" method,
        # don't trust scipy.optimize.minimize documentation.

        optimization_result = minimize(
            fun=compute_invgauss_negloglikel,
            x0=params0,
            method="trust-constr",
            jac=compute_invgauss_negloglikel_grad,
            hess=compute_invgauss_negloglikel_hessian,
            constraints=cons,
            args=(self.dataset.xn, self.dataset.wn, self.dataset.eta),
            options={"maxiter": self.max_steps, "disp": False},
            callback=_save_history,
        )
        print(f"Number of iterations: {optimization_result.nit}")
        print(
            f"Optimization process outcome: {'Success' if optimization_result.success else 'Failed'}"
        )
        optimal_parameters = optimization_result.x
        k_param, thetap_params = (
            optimal_parameters[0],
            optimal_parameters[1 : 1 + self.m],
        )

        return build_ig_model(
            thetap_params,
            k_param,
            self.dataset.wn,
            self.dataset.hasTheta0,
            results,
            params_history,
        )
