from abc import ABC
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from pp.core.distributions.inverse_gaussian import (
    InverseGaussianConstraints,
    compute_invgauss_negloglikel,
    compute_invgauss_negloglikel_grad,
    compute_invgauss_negloglikel_hessian,
    compute_lambda,
    likel_invgauss_consistency_check,
)
from pp.model import PointProcessDataset, PointProcessMaximizer, PointProcessResult


class InverseGaussianMaximizer(PointProcessMaximizer, ABC):
    def __init__(
        self,
        dataset: PointProcessDataset,
        max_steps: int = 200,
        theta0: Optional[np.array] = None,
        k0: Optional[float] = None,
        verbose: bool = True,
        save_history: bool = False,
    ):
        """
            Args:
                dataset: PointProcessDataset to use for the regression.
                max_steps: max_steps is the maximum number of allowed iterations of the optimization process.
                theta0: is a vector of shape (p,1) (or (p+1,1) if teh dataset was created with the hasTheta0 option)
                 of coefficients used as starting point for the optimization process.
                k0: is the starting point for the scale parameter (sometimes called lambda).
                verbose: If True convergence information will be displayed
                save_history: If True the PointProcessResult returned by the train() routine will contain additional / useful
                              information about the training process. (Check the definition of PointProcessResult for details)
            Returns:
                PointProcessModel
            """
        self.dataset = dataset
        self.max_steps = max_steps
        self.theta0 = theta0
        self.k0 = k0
        self.n, self.m = self.dataset.xn.shape
        self.verbose = verbose
        self.save_history = save_history
        # Some consistency checks
        likel_invgauss_consistency_check(
            self.dataset.xn, self.dataset.wn, self.dataset.xt, self.theta0
        )

    def train(self) -> PointProcessResult:
        """

        Info:
            This function implements the optimization process suggested by Riccardo Barbieri, Eric C. Matten,
            Abdul Rasheed A. Alabi. and Emery N. Brown in the paper:
            "A point-process model of human heartbeat intervals: new definitions
            of heart rate and heart rate variability"

        """

        params_history = []
        results = []

        def _save_history(params: np.array, state):  # pragma: no cover
            results.append(
                compute_invgauss_negloglikel(
                    params, self.dataset.xn, self.dataset.wn, self.dataset.eta
                )
            )
            params_history.append(params)

        # TODO change initialization (maybe?)
        if self.theta0 is None:
            self.theta0 = np.ones((self.m, 1)) / self.m
        if self.k0 is None:
            self.k0 = 1700

        # In order to optimize the parameters with scipy.optimize.minimize we need to pack all of our parameters in a
        # vector of shape (1+p,) or (1+1+p,) if hasTheta0
        params0 = np.vstack((self.k0, self.theta0)).squeeze(1)

        cons = InverseGaussianConstraints(self.dataset.xn)()
        # it's ok to have cons as a list of LinearConstrainsts if we're using the "trust-constr" method,
        # don't trust scipy.optimize.minimize documentation.
        # decrease tol to observe the "right-censoring ridges"
        optimization_result = minimize(
            fun=compute_invgauss_negloglikel,
            x0=params0,
            method="trust-constr",
            jac=compute_invgauss_negloglikel_grad,
            hess=compute_invgauss_negloglikel_hessian,
            constraints=cons,
            tol=0.0125,
            args=(
                self.dataset.xn,
                self.dataset.wn,
                self.dataset.eta,
                self.dataset.xt,
                self.dataset.wt,
            ),
            options={"maxiter": self.max_steps, "disp": False},
            callback=_save_history if self.save_history else None,
        )

        if not optimization_result.success:  # pragma: no cover
            print(
                f"\nWarning: Convergence not reached at time bin {self.dataset.current_time}\n"
            )

        if self.verbose:  # pragma: no cover
            print(
                f"\rNumber of iterations: {optimization_result.nit}\n"
                f"Optimization process outcome: {'Success' if optimization_result.success else 'Failed'}"
            )
        optimal_parameters = optimization_result.x
        k_param, thetap_params = (
            optimal_parameters[0],
            optimal_parameters[1 : 1 + self.m],
        )

        # Compute prediction
        mu = np.dot(self.dataset.xt, thetap_params.reshape(-1, 1))[0, 0]
        # Compute sigma
        sigma = mu ** 3 / k_param

        # Compute Lambda (Hazard Function)
        # for the current_time
        if self.dataset.wt is not None:  # pragma: no cover
            lambda_ = compute_lambda(mu, k_param, self.dataset.wt)
        else:
            lambda_ = None

        return PointProcessResult(
            self.dataset.hasTheta0,
            thetap_params,
            k_param,
            self.dataset.current_time,
            mu,
            sigma,
            np.mean(self.dataset.wn),
            lambda_,
            self.dataset.target,
            results if results else None,
            np.hstack([params.reshape(-1, 1) for params in params_history])
            if params_history
            else None,
        )
