from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable

import numpy as np
from scipy.linalg import toeplitz


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
        distribution: InterEventDistribution,
        ar_order: int,
        hasTheta0: bool,
    ):
        self._model = model
        self.expected_input_shape = expected_shape
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

    def __call__(self, *args, **kwargs):
        return self._model(*args)


class PointProcessDataset:
    def __init__(self, xn: np.ndarray, wn: np.ndarray, p: int, hasTheta0: bool):
        self.xn = xn
        self.wn = wn
        self.p = p
        self.hasTheta0 = hasTheta0

    def __repr__(self):
        return f"<PointProcessDataset: <xn.shape={self.xn.shape}> <wn.shape={self.wn.shape}> <hasTheta0={self.hasTheta0}>>"

    @classmethod
    def load(cls, inter_event_times: np.ndarray, p: int, hasTheta0: bool = True):
        """

        Args:
            inter_event_times: np.ndarray of inter-events times expressed in ms.
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
        wn = inter_event_times[p:]
        # We prefer to column vector of shape (m,1) instead of row vector of shape (m,)
        wn = wn.reshape(-1, 1)

        # We now have to build a matrix xn s.t. for i = 0, ..., len(rr)-p-1 the i_th element of xn will be
        # xn[i] = [1, rr[i + p - 1], rr[i + p - 2], ..., rr[i]]
        # Note that the 1 at the beginning of each row is added only if the hasTheta0 parameter is set to True.
        a = inter_event_times[p - 1 : -1]
        b = inter_event_times[p - 1 :: -1]
        xn = toeplitz(a, b)
        if hasTheta0:
            xn = np.hstack([np.ones(wn.shape), xn])
        return cls(xn, wn, p, hasTheta0)


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
