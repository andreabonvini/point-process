from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, List, Union

import numpy as np
from scipy.linalg import toeplitz
from scipy.optimize import LinearConstraint, NonlinearConstraint


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

    def __call__(self, inter_event_times: np.ndarray):
        return self._model(inter_event_times)


class PointProcessDataset:
    def __init__(self, xn: np.ndarray, wn: np.ndarray, p: int, hasTheta0: bool):
        self.xn = xn
        self.wn = wn
        self.p = p
        self.hasTheta0 = hasTheta0

    def __repr__(self):
        return f"<PointProcessDataset: <xn.shape={self.xn.shape}> <wn.shape={self.wn.shape}> <hasTheta0={self.hasTheta0}>>"

    @classmethod
    def load(cls, inter_events_times: np.ndarray, p: int, hasTheta0: bool = True):
        """

        Args:
            inter_events_times: np.ndarray of inter-events times expressed in ms.
            p: AR order
            hasTheta0: whether or not the AR model has a theta0 constant to account for the average mu.

        Returns:
            PointProcessDataset where:
                xn.shape : (len(events)-p,p) or (len(events)-p,p+1) if hasTheta0 is set to True.
                wn-shape : (len(events)-p,1).
                each row of xn is associated to the corresponding element of wn.
        """
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
        if hasTheta0:
            xn = np.hstack([np.ones(wn.shape), xn])
        return cls(xn, wn, p, hasTheta0)


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


class WeightsProducer(ABC):  # pragma: no cover
    # FIXME mypy fails if abstract __call__ is defined
    # @abstractmethod
    # def __call__(self, *args, **kwargs) -> np.ndarray:
    #   return self._compute_weights()

    @abstractmethod
    def _compute_weights(self) -> np.ndarray:
        pass
