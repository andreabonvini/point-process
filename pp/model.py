from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List, Optional, Union

import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import LinearConstraint, NonlinearConstraint

from pp import ExponentialWeightsProducer


class InterEventDistribution(Enum):
    INVERSE_GAUSSIAN = "Inverse Gaussian"


class PointProcessResult:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return f"mu: {self.mu}\nsigma: {self.sigma}"


class PointProcessModel:
    def __init__(
        self,
        model: Callable[[np.ndarray], PointProcessResult],
        expected_shape: tuple,
        theta: np.ndarray,
        k: float,
        results: List[float],
        params_history: List[np.ndarray],
        distribution: InterEventDistribution,
        wn: np.ndarray,
        ar_order: int,
        hasTheta0: bool,
    ):
        """
        Args:
            model: actual model which yields a PointProcessResult
            expected_shape: expected input shape to feed the PointProcessModel with
            theta: final AR parameters.
            k: final shape parameter (aka lambda).
            results: negative log-likelihood values obtained during the optimization process (should diminuish in time).
            params_history: list of parameters obtained during the optimization process
            distribution: fitting distribution used to train the model.
            wn: target inter-events times used to train the model.
            ar_order: AR order used to train the model
            hasTheta0: if the model was trained with theta0 parameter
        """
        self._model = model
        self.expected_input_shape = expected_shape
        self.theta = theta
        self.k = k
        self.results = results
        self.params_history = params_history
        self.distribution = distribution
        self.wn = wn
        self.ar_order = ar_order
        self.hasTheta0 = hasTheta0

    def __repr__(self):
        return (
            f"<PointProcessModel<\n"
            f"\t<model={self._model}>\n"
            f"\t<expected_input_shape={self.expected_input_shape}>\n"
            f"\t<distributuon={self.distribution}>\n"
            f"\t<ar_order={self.ar_order}>\n"
            f"\t<hasTheta0={self.hasTheta0}>\n"
            f">"
        )

    def __call__(self, inter_event_times: np.ndarray) -> PointProcessResult:
        return self._model(inter_event_times)


class PointProcessDataset:
    def __init__(
        self,
        xn: np.ndarray,
        wn: np.ndarray,
        p: int,
        hasTheta0: bool,
        eta: np.array,
        xt: Optional[np.array] = None,
        wt: Optional[float] = None,
    ):
        """

        Args:
            xn:
            wn: target values for the given dataset
            p: AR order
            hasTheta0: whether or not the AR model has a theta0 constant to account for the average mu.
            eta: weights for each sample.
            xt: is a vector 1xN of regressors, for the censoring part. (IF RIGHT-CENSORING)  # FIXME what's N
            wt: is the current value of the future observation. (IF RIGHT-CENSORING) # FIXME which one is right?
            wt: if right_censoring is applied, wt represents the distance (in seconds) between the last observed
            event and the current evaluation time (evaluation time > last observed event) # FIXME which one is right?
        """
        self.xn = xn
        self.wn = wn
        self.p = p
        self.hasTheta0 = hasTheta0
        self.eta = eta
        self.xt = xt
        self.wt = wt
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
    ):
        """

        Args:
            event_times: np.ndarray of event times expressed in s.
            p: AR order
            hasTheta0: whether or not the AR model has a theta0 constant to account for the average mu.
            weights_producer: WeightsProducer object
            right_censoring: whether right-censoring is applied or not
            current_time: if right-censoring is applied, the current time at which we are evaluating our model
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
        if right_censoring:
            xt = inter_events_times[-p:][::-1].reshape(1, -1)
            xt = np.hstack([[[1.0]], xt]) if hasTheta0 else xt
            # FIXME remove
            wt = current_time - event_times[-1]
        else:
            xt = wt = None
        xn = toeplitz(a, b)
        xn = np.hstack([np.ones(wn.shape), xn]) if hasTheta0 else xn
        uk = event_times[p + 1 :]
        if current_time is None:
            # In case current_time was not provided we suppose we are evaluating our model at t = last observed event
            current_time = event_times[-1]

        eta = weights_producer(current_time - uk)

        return cls(xn, wn, p, hasTheta0, eta, xt, wt)


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
    def train(self) -> PointProcessModel:
        pass
