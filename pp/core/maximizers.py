from abc import ABC
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from pp.core.model import (
    PointProcessDataset,
    PointProcessMaximizer,
    PointProcessModel,
    WeightsProducer,
)
from pp.core.utils.constraints import greater_than_zero
from pp.core.utils.inverse_gaussian import (
    build_ig_model,
    compute_invgauss_negloglikel,
    compute_invgauss_negloglikel_grad,
    compute_invgauss_negloglikel_hessian,
    likel_invgauss_consistency_check,
)
from pp.core.weights_producers import (
    ConstantWeightsProducer,
    ExponentialWeightsProducer,
)


class InverseGaussianMaximizer(PointProcessMaximizer, ABC):
    def __init__(
        self,
        dataset: PointProcessDataset,
        max_steps: int = 1000,
        weights_producer: WeightsProducer = ConstantWeightsProducer(),
        thetap0: Optional[np.array] = None,
        k0: Optional[float] = None,
        xt: Optional[np.array] = None,
        wt: Optional[float] = None,
    ):
        """
            Args:
                dataset: PointProcessDataset to use for the regression.
                max_steps: max_steps is the maximum number of allowed iterations of the optimization process.
                weights_producer: WeightsProducer object.
                thetap0: is a vector Nx1 of coefficients used as starting point for the optimization process.
                k0: is the starting point for the scale parameter (sometimes called lambda).
                xt: is a vector 1xN of regressors, for the censoring part. (IF RIGHT-CENSORING)
                wt: is the current value of the future observation. (IF RIGHT-CENSORING)
            Returns:
                PointProcessModel
            """
        self.dataset = dataset
        self.max_steps = max_steps
        self.thetap0 = thetap0
        self.k0 = k0
        self.xt = xt
        self.wt = wt
        self.m, self.n = self.dataset.xn.shape
        # Some consistency checks
        likel_invgauss_consistency_check(
            self.dataset.xn, self.dataset.wn, self.xt, self.thetap0
        )
        # weights initialization
        if isinstance(weights_producer, ConstantWeightsProducer):
            self.eta = weights_producer(self.m)
        elif isinstance(
            weights_producer, ExponentialWeightsProducer
        ):  # pragma: no cover
            # TODO test it and parametrize alpha!
            self.eta = weights_producer(target_intervals=self.dataset.wn, alpha=0.1)

    def train(self) -> PointProcessModel:

        # TODO change initialization
        if self.thetap0 is None:
            self.thetap0 = np.ones((self.n, 1)) / self.n
        if self.k0 is None:
            self.k0 = 0.5

        # In order to optimize the parameters with scipy.optimize.minimize we need to pack all of our parameters in a
        # vector of shape (1+n,)
        params0 = np.vstack([self.k0, self.thetap0]).squeeze(1)

        # we want our parameters to be greater than 0
        cons = greater_than_zero(self.n + 1)

        optimization_result = minimize(
            fun=compute_invgauss_negloglikel,
            x0=params0,
            jac=compute_invgauss_negloglikel_grad,
            hess=compute_invgauss_negloglikel_hessian,
            method="trust-constr",
            args=(self.dataset.xn, self.dataset.wn),
            constraints=cons,
            options={"maxiter": self.max_steps, "disp": True},
        )
        optimal_parameters = optimization_result.x
        k_param, thetap_params = (
            optimal_parameters[0],
            optimal_parameters[1 : 1 + self.n],
        )
        return build_ig_model(thetap_params, k_param, self.dataset.hasTheta0)
