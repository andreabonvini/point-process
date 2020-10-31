import numpy as np

from pp.core.distributions import events2interevents
from pp.core.maximizers import InverseGaussianMaximizer
from pp.core.model import InterEventDistribution, PointProcessDataset, PointProcessModel

maximizers_dict = {
    InterEventDistribution.INVERSE_GAUSSIAN.value: InverseGaussianMaximizer
}


def regr_likel(
    events: np.ndarray,
    maximizer_distribution: InterEventDistribution,
    p: int = 9,
    hasTheta0: bool = True,
) -> PointProcessModel:
    """
        @param events:
            event-times as returned by the pp.utils.load() function.
            MUST be expressed in <seconds>.
        @param maximizer_distribution:
            log-likelihood maximization function belonging to the Maximizer enum.
        @param p:
            auto-regressive order.
        @param hasTheta0:
             whether or not the AR model has a theta0 constant to account for the average mu.

        @return:
            PointProcessModel for the given configuration.

    """
    inter_event_times = events2interevents(events)

    dataset = PointProcessDataset.load(inter_event_times, p, hasTheta0)

    return maximizers_dict[maximizer_distribution.value](dataset).train()
