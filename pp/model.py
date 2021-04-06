from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from scipy.linalg import toeplitz

from pp import ExponentialWeightsProducer


class InterEventDistribution(Enum):
    INVERSE_GAUSSIAN = "Inverse Gaussian"


@dataclass
class InverseGaussianResult:
    """
        Args:
            theta: final AR parameters.
            k: final shape parameter (aka lambda).
            current_time: current evaluatipon time
            mu: final mu prediction for current_time.
            sigma: final sigma prediction for current_time.
            mean_interval: mean target interval, it is useful just to compute the spectral components
            target: (Optional) expected mu prediction for current_time
    """

    theta: np.ndarray
    k: float
    current_time: float
    mu: float
    sigma: float
    mean_interval: float
    target: Optional[float] = None


class PointProcessDataset:
    def __init__(
        self,
        xn: np.ndarray,
        wn: np.ndarray,
        p: int,
        eta: np.ndarray,
        current_time: float,
        xt: np.ndarray,
        target: Optional[float] = None,
        wt: Optional[float] = None,
    ):
        """
        Args:
            xn: lagged time intervals
            wn: target values for the given dataset
            p: AR order
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
        self.eta = eta
        self.current_time = current_time
        self.xt = xt
        self.target = target
        if wt is not None:
            assert wt >= 0
        else:
            wt = 0.0
        self.wt = wt

    def __repr__(self):
        return (
            f"<PointProcessDataset:\n"
            f"    <xn.shape={self.xn.shape}>\n"
            f"    <wn.shape={self.wn.shape}>\n"
            f">"
        )

    @classmethod
    def load(
        cls,
        event_times: np.ndarray,
        p: int,
        weights_producer: ExponentialWeightsProducer = ExponentialWeightsProducer(),
        current_time: Optional[float] = None,
        target: Optional[float] = None,
    ):
        """

        Args:
            event_times: np.ndarray of event times expressed in s.
            p: AR order
            weights_producer: WeightsProducer object
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
        xn = np.hstack([np.ones(wn.shape), xn])

        xt = inter_events_times[-p:][::-1].reshape(1, -1)
        xt = np.hstack([[[1.0]], xt])

        if current_time is None:
            # In case current_time was not provided we suppose we are evaluating our model at t = last observed event
            current_time = event_times[-1]

        wt = current_time - event_times[-1]

        uk = event_times[p + 1 :]

        eta = weights_producer(current_time - uk)
        return cls(xn, wn, p, eta, current_time, xt, target, wt)
