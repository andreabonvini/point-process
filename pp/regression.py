from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from pp.core.maximizers import InverseGaussianMaximizer
from pp.model import InterEventDistribution, PointProcessDataset, PointProcessModel

maximizers_dict = {
    InterEventDistribution.INVERSE_GAUSSIAN.value: InverseGaussianMaximizer
}


@dataclass
class PipelineResult:
    thetaps: List[np.array]
    ks: List[float]
    times: List[float]
    mus: List[float]


def regr_likel(
    dataset: PointProcessDataset,
    maximizer_distribution: InterEventDistribution,
    theta0: Optional[np.array] = None,
    k0: Optional[float] = None,
) -> PointProcessModel:
    """
    Args:
        dataset: PointProcessDataset containing the specified AR order (p)
        and hasTheta0 option (if we want to account for the bias)
        maximizer_distribution: log-likelihood maximization function belonging to the Maximizer enum.
        theta0: starting vector for the theta parameters
        k0: starting value for the k parameter

    Returns:
        Trained PointProcessModel
    """

    return maximizers_dict[maximizer_distribution.value](
        dataset=dataset, theta0=theta0, k0=k0
    ).train()


def _pipeline_setup(
    event_times: np.array, window_length: float, delta: float
) -> Tuple[int, int, int]:
    # Firstly some consistency check
    if event_times[-1] < window_length:
        raise Exception(
            ValueError(
                f"The window length is too wide (window_length:{str(window_length)}), the "
                f"inter event times provided has a total cumulative length "
                f"{event_times[-1]} < {str(window_length)}"
            )
        )
    # Find the index of the last event within window_length
    last_event_index = np.where(event_times > window_length)[0][0] - 1
    # Find total number of time bins
    bins = int(event_times[-1] // delta) + 1
    # We have to ignore the first window since we have to use it for initialization purposes,
    # we find the number of time bins contained in one window and start our regression process from there.
    bins_in_window = int(np.ceil(window_length / delta))
    return last_event_index, bins, bins_in_window


def regr_likel_pipeline(
    event_times: np.array,
    ar_order: int = 9,
    hasTheta0: bool = True,
    window_length: float = 60.0,
    delta: float = 0.005,
) -> PipelineResult:
    """
    Args:
        event_times: event times expressed in seconds.
        ar_order: AR order to use in the regression process
        hasTheta0: whether or not the AR model has a theta0 constant to account for the average mu.
        window_length: time window used for local likelihood maximization.
        delta: how much the local likelihood time interval is shifted to compute the next parameter update,
        be careful: time_resolution must be little enough s.t. at most ONE event can happen in each time bin.

    Returns:
        a PipelineResult object.
    """
    # We want the first event to be at time 0
    events = event_times - event_times[0]
    # last_event_index is the index of the last event within the first window
    # e.g. if events = [0.0, 1.3, 2.1, 3.2, 3.9, 4.5] and window_length = 3.5 then last_event_index = 3
    # bins is the total number of bins we can discretize our events with (given our time_resolution)
    last_event_index, bins, bins_in_window = _pipeline_setup(
        events, window_length, delta
    )
    # observed_events here is the subset of events observed during the first window, this np.array will keep track
    # of the events used for local regression at each time bin, discarding old events and adding new ones.
    # It works as a buffer for our regression process.
    observed_events = events[: last_event_index + 1]  # +1 since we want to include it!
    # Initialize model parameters to None
    thetap = None
    k = None
    # Initialize result lists
    all_thetas = []
    all_ks = []
    all_times = []
    all_mus = []
    for bin_index in range(
        bins_in_window, bins + 1
    ):  # bins + 1 since we want to include the last one!
        # current_time is the time (expressed in seconds) associated with the given bin.
        current_time = bin_index * delta
        # If the first element of observed_events happened before the
        # time window between (current_time - window_length) and (current_time)
        # we can discard it since it will not be part of the current optimization process.
        if (
            observed_events.size > 0
            and observed_events[0] < current_time - window_length
        ):
            # Remove older event (there could be only one because we assume that
            # in any delta interval (aka time_resolution) there is at most one event)
            observed_events = np.delete(observed_events, 0, 0)  # remove first element
            # Force re-evaluation of starting point for thetap
            thetap = None
        # We check whether an event happened in ((bin_index - 1) * time_resolution, bin_index * time_resolution]
        event_happened: bool = events[last_event_index + 1] <= current_time
        if event_happened:
            last_event_index += 1
            # Append current event
            observed_events = np.append(observed_events, events[last_event_index])
            # Force re-evaluation of starting point for thetap
            thetap = None
        # Now if thetap is empty (i.e., observed_events has changed), re-evaluate the
        # variables that depend on observed_events
        if thetap is None:
            dataset = PointProcessDataset.load(
                event_times=observed_events, p=ar_order, hasTheta0=hasTheta0,
            )
            # The uncensored log-likelihood is a good starting point
            model = regr_likel(dataset, InterEventDistribution.INVERSE_GAUSSIAN)
            thetap, k = model.theta, model.k
        else:
            # If we end up in this branch, then the only thing that change is the right-censoring part,
            # we can use the thetap and k computed in the previous iteration as starting point for the optimization
            # process.
            pass
        dataset = PointProcessDataset.load(
            event_times=observed_events,
            p=ar_order,
            hasTheta0=hasTheta0,
            right_censoring=True,
            current_time=current_time,
        )
        model = regr_likel(
            dataset=dataset,
            maximizer_distribution=InterEventDistribution.INVERSE_GAUSSIAN,
            theta0=thetap.reshape(-1, 1),
            k0=k,
        )
        mu = np.dot(dataset.xt, model.theta.reshape(-1, 1))[0, 0]
        all_thetas.append(deepcopy(model.theta))
        all_ks.append(deepcopy(model.k))
        all_times.append(current_time)
        all_mus.append(mu)
    return PipelineResult(all_thetas, all_ks, all_times, all_mus)
