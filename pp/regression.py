import numpy as np
from scipy.linalg import toeplitz

from pp.core.maximizers import inverse_gaussian_maximizer
from pp.core.model import InterEventDistribution, PointProcessModel

maximizers_dict = {
    InterEventDistribution.INVERSE_GAUSSIAN.value: inverse_gaussian_maximizer
}


def regr_likel(
    events: np.ndarray,
    maximizer: InterEventDistribution,
    p: int = 9,
    hasTheta0: bool = True,
) -> PointProcessModel:
    """
        @param events:
            event-times as returned by the pp.utils.load() function.
        @param maximizer:
            log-likelihood maximization function belonging to the Maximizer enum.
        @param p:
            auto-regressive order.
        @param hasTheta0:
             whether or not the AR model has a theta0 constant to account for the average mu.

        @return:
            PointProcessModel for the given configuration.

    """

    # We reset the events s.t. the first event is at time 0.
    observ_ev = events - events[0]

    # rr is a np.array which contains the inter-event intervals expressed in ms.
    rr = np.diff(observ_ev) * 1000
    # wn are the target inter-event intervals, i.e. the intervals we have to predict once we build our
    # RR autoregressive model.
    wn = rr[p:]
    # We prefer to column vector of shape (m,1) instead of row vector of shape (m,)
    wn = wn.reshape(-1, 1)

    # We now have to build a matrix xn s.t. for i = 0, ..., len(rr)-p-1 the i_th element of xn will be
    # xn[i] = [1, rr[i + p - 1], rr[i + p - 2], ..., rr[i]]
    # Note that the 1 at the beginning of each row is added only if the hasTheta0 parameter is set to True.
    a = rr[p - 1 : -1]
    b = rr[p - 1 :: -1]
    xn = toeplitz(a, b)
    if hasTheta0:
        xn = np.hstack([np.ones(wn.shape), xn])

    return maximizers_dict[maximizer.value](xn, wn)
