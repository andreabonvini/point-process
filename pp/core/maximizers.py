from typing import Optional

import numpy as np

import pp.optimized.py_regr_likel as opt_rl
from pp.core.distributions.inverse_gaussian import likel_invgauss_consistency_check
from pp.model import InverseGaussianResult, PointProcessDataset


class InverseGaussianMaximizer:
    def __init__(
        self,
        dataset: PointProcessDataset,
        max_steps: int,
        theta0: Optional[np.ndarray] = None,
        k0: Optional[float] = None,
        right_censoring: bool = False,
        do_global: bool = False,
    ):
        """
            Args:
                dataset: PointProcessDataset to use for the regression.
                max_steps: max_steps is the maximum number of allowed iterations of the optimization process.
                theta0: is a vector of shape (p,1) (or (p+1,1) if teh dataset was created with the hasTheta0 option)
                 of coefficients used as starting point for the optimization process.
                k0: is the starting point for the scale parameter (sometimes called lambda).
                right_censoring: True if we want to apply right censoring.
                do_global: True if we want to initialize the parameters through a global optimization procedure.
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
        self.right_censoring = right_censoring
        self.do_global = do_global

    def train(self) -> InverseGaussianResult:
        """

        Info:
            This function just calls a c-function which implements the optimization process suggested by Riccardo Barbieri, Eric C. Matten,
            Abdul Rasheed A. Alabi. and Emery N. Brown in the paper:
            "A point-process model of human heartbeat intervals: new definitions
            of heart rate and heart rate variability"
            Check the file c_reg_likel.c for more details.
        Returns:
            an Inverse Gaussian Result, this object will likely be used to save data to a .csv file.

        """

        mean_interval = float(
            np.dot(self.dataset.eta.T, self.dataset.wn) / np.sum(self.dataset.eta)
        )
        # TODO change initialization (maybe?)
        if self.theta0 is None:
            self.theta0 = np.ones((self.m, 1)) * 0.001
            self.theta0[0] = mean_interval
        if self.k0 is None:
            self.k0 = 1500.0

        xn = self.dataset.xn
        eta = self.dataset.eta
        wn = self.dataset.wn
        xt = self.dataset.xt
        wt = self.dataset.wt

        params = opt_rl.regr_likel(
            self.dataset.p,
            self.n,
            self.max_steps,
            self.theta0,
            self.k0,
            xn,
            eta,
            wn,
            xt,
            wt,
            self.right_censoring,
            self.do_global,
        )
        k = params[0]
        thetap = params[1:]

        # Compute prediction
        mu = np.dot(xt, thetap.reshape(-1, 1))[0, 0]
        # Compute sigma
        sigma = mu ** 3 / k

        return InverseGaussianResult(
            thetap,
            k,
            self.dataset.current_time,
            mu,
            sigma,
            mean_interval,
            self.dataset.target,
        )
