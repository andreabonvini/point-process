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
from pp.core.model import (
    PointProcessDataset,
    PointProcessMaximizer,
    PointProcessModel,
    WeightsProducer,
)
from pp.core.weights_producers import (
    ConstantWeightsProducer,
    ExponentialWeightsProducer,
)

params_history = []
results = []


class InverseGaussianMaximizer(PointProcessMaximizer, ABC):
    def __init__(
        self,
        dataset: PointProcessDataset,
        weights_producer: WeightsProducer,
        max_steps: int = 1000,
        theta0: Optional[np.array] = None,
        k0: Optional[float] = None,
        xt: Optional[np.array] = None,
        wt: Optional[float] = None,
    ):
        """
            Args:
                dataset: PointProcessDataset to use for the regression.
                max_steps: max_steps is the maximum number of allowed iterations of the optimization process.
                weights_producer: WeightsProducer object.
                theta0: is a vector of shape (p,1) (or (p+1,1) if teh dataset was created with the hasTheta0 option)
                 of coefficients used as starting point for the optimization process.
                k0: is the starting point for the scale parameter (sometimes called lambda).
                xt: is a vector 1xN (#FIXME what's N) of regressors, for the censoring part. (IF RIGHT-CENSORING)
                wt: is the current value of the future observation. (IF RIGHT-CENSORING)
            Returns:
                PointProcessModel
            """
        self.dataset = dataset
        self.max_steps = max_steps
        self.theta0 = theta0
        self.k0 = k0
        self.xt = xt
        self.wt = wt
        self.n, self.m = self.dataset.xn.shape
        # Some consistency checks
        likel_invgauss_consistency_check(
            self.dataset.xn, self.dataset.wn, self.xt, self.theta0
        )
        # weights initialization
        if isinstance(weights_producer, ConstantWeightsProducer):  # pragma: no cover
            self.eta = weights_producer(self.n)
        elif isinstance(
            weights_producer, ExponentialWeightsProducer
        ):  # pragma: no cover
            self.eta = weights_producer(target_intervals=self.dataset.wn)

    def train(self) -> PointProcessModel:

        global params_history
        global results

        # TODO change initialization
        if self.theta0 is None:
            self.theta0 = np.ones((self.m, 1)) / self.m
        if self.k0 is None:
            self.k0 = 1200

        # In order to optimize the parameters with scipy.optimize.minimize we need to pack all of our parameters in a
        # vector of shape (1+p,) or (1+1+p,) if hasTheta0
        params0 = np.vstack([self.k0, self.theta0]).squeeze(1)

        def _save_history(params: np.array, state):  # pragma: no cover
            global params_history
            global results
            results.append(
                compute_invgauss_negloglikel(
                    params, self.dataset.xn, self.dataset.wn, self.eta
                )
            )
            params_history.append(params)

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
            args=(self.dataset.xn, self.dataset.wn, self.eta),
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
            thetap_params, k_param, self.dataset.hasTheta0, results, params_history
        )
