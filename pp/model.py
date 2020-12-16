from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import LinearConstraint, NonlinearConstraint

from pp import ExponentialWeightsProducer


class InterEventDistribution(Enum):
    INVERSE_GAUSSIAN = "Inverse Gaussian"


@dataclass
class PointProcessResult:
    """
        Args:
            hasTheta0: if the model was trained with theta0 parameter
            theta: final AR parameters.
            k: final shape parameter (aka lambda).
            current_time: current evaluatipon time
            mu: final mu prediction for current_time.
            sigma: final sigma prediction for current_time.
            mean_interval: mean target interval, it is useful just to compute the spectal components
            lambda_: Inhomogenous Poisson rate (or hazard function) at current_time
            target: (Optional) expected mu prediction for current_time
            results:  (Optional)
                negative log-likelihood values obtained during the optimization process (should diminuish in time).
            params_history: (Optional)
                matrix containing the values obtained for each parameter at each iteration of the optimization
                process, every row represent a different parameter.
                If the trained model distribution was an Inverse Gaussian the rows represent k,theta0,...,theatap
    """

    hasTheta0: bool
    theta: np.ndarray
    k: float
    current_time: float
    mu: float
    sigma: float
    mean_interval: float
    lambda_: float
    target: Optional[float] = None
    results: Optional[List[float]] = None
    params_history: Optional[List[np.ndarray]] = None


class PointProcessDataset:
    def __init__(
        self,
        xn: np.ndarray,
        wn: np.ndarray,
        p: int,
        hasTheta0: bool,
        eta: np.array,
        current_time: float,
        xt: np.array,
        target: Optional[float] = None,
        wt: Optional[float] = None,
    ):
        """

        Args:
            xn: lagged time intervals
            wn: target values for the given dataset
            p: AR order
            hasTheta0: whether or not the AR model has a theta0 constant to account for the average mu.
            eta: weights for each sample.
            xt: is a vector 1xp (or 1x(p+1) ) of regressors, for the censoring part.
            wt: is the current value of the future observation. (IF RIGHT-CENSORING)
                If right_censoring is applied, wt represents the distance (in seconds) between the last observed
                 event and the current evaluation time (evaluation time > last observed event)
            target: target time interval for the current time bin
        """
        self.xn = xn
        self.wn = wn
        self.p = p
        self.hasTheta0 = hasTheta0
        self.eta = eta
        self.current_time = current_time
        self.xt = xt
        self.wt = wt
        self.target = target
        if wt is not None:
            assert wt >= 0

    def __repr__(self):
        return (
            f"<PointProcessDataset:\n"
            f"    <xn.shape={self.xn.shape}>\n"
            f"    <wn.shape={self.wn.shape}>\n"
            f"    <hasTheta0={self.hasTheta0}>>"
        )

    @classmethod
    def load(
        cls,
        event_times: np.ndarray,
        p: int,
        hasTheta0: bool = True,
        weights_producer: ExponentialWeightsProducer = ExponentialWeightsProducer(),
        right_censoring: bool = False,
        current_time: Optional[float] = None,
        target: Optional[float] = None,
    ):
        """

        Args:
            event_times: np.ndarray of event times expressed in s.
            p: AR order
            hasTheta0: whether or not the AR model has a theta0 constant to account for the average mu.
            weights_producer: WeightsProducer object
            right_censoring: whether right-censoring is applied or not
            current_time: if right-censoring is applied, the current time at which we are evaluating our model
            target: target time interval for the current time bin
        """
        inter_events_times = np.diff(event_times)
        # wn are the target inter-event intervals, i.e. the intervals we have to predict once we build our
        # RR autoregressive model.
        wn = inter_events_times[p:]
        # We prefer to column vector of shape (m,1) instead of row vector of shape (m,)
        wn = wn.reshape(-1, 1)

        # We now have to build a matrix xn s.t. for i = 0, ..., len(rr)-p-1 the i_th element of xn will be
        # xn[i] = [1, rr[i + p - 1], rr[i + p - 2], ..., rr[i]]
        # Note that the 1 at the beginning of each row is added only if the hasTheta0 parameter is set to True.
        a = inter_events_times[p - 1 : -1]
        b = inter_events_times[p - 1 :: -1]
        xn = toeplitz(a, b)
        xn = np.hstack([np.ones(wn.shape), xn]) if hasTheta0 else xn

        xt = inter_events_times[-p:][::-1].reshape(1, -1)
        xt = np.hstack([[[1.0]], xt]) if hasTheta0 else xt

        wt = current_time - event_times[-1] if right_censoring else None

        uk = event_times[p + 1 :]

        if current_time is None:
            # In case current_time was not provided we suppose we are evaluating our model at t = last observed event
            current_time = event_times[-1]

        eta = weights_producer(current_time - uk)

        return cls(xn, wn, p, hasTheta0, eta, current_time, xt, target, wt)


class PointProcessConstraint(ABC):  # pragma: no cover
    @abstractmethod
    def __call__(self) -> List[Union[LinearConstraint, NonlinearConstraint]]:
        return self._compute_constraints()

    @abstractmethod
    def _compute_constraints(
        self,
    ) -> List[Union[LinearConstraint, NonlinearConstraint]]:
        pass


class PointProcessMaximizer(ABC):  # pragma: no cover
    @abstractmethod
    def train(self) -> PointProcessResult:
        pass
