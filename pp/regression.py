from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from pp.core.distributions.inverse_gaussian import compute_lambda, igcdf
from pp.core.maximizers import InverseGaussianMaximizer
from pp.model import InterEventDistribution, PointProcessDataset, PointProcessResult

maximizers_dict = {
    InterEventDistribution.INVERSE_GAUSSIAN.value: InverseGaussianMaximizer
}


def regr_likel(
    dataset: PointProcessDataset,
    maximizer_distribution: InterEventDistribution,
    theta0: Optional[np.array] = None,
    k0: Optional[float] = None,
    verbose: bool = False,
    save_history: bool = False,
) -> PointProcessResult:
    """
    Args:
        dataset: PointProcessDataset containing the specified AR order (p)
        and hasTheta0 option (if we want to account for the bias)
        maximizer_distribution: log-likelihood maximization function belonging to the Maximizer enum.
        theta0: starting vector for the theta parameters
        k0: starting value for the k parameter
        verbose: If True convergence information will be displayed
        save_history: If True the PointProcessResult returned by the train() routine will contain additional / useful
                      information about the training process. (Check the definition of PointProcessResult for details)

    Returns:
        PointProcessResult
    """

    return maximizers_dict[maximizer_distribution.value](
        dataset=dataset,
        theta0=deepcopy(theta0),
        k0=deepcopy(k0),
        verbose=verbose,
        save_history=save_history,
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
    bins = int(np.ceil(event_times[-1] / delta))
    # We have to ignore the first window since we have to use it for initialization purposes,
    # we find the number of time bins contained in one window and start our regression process from there.
    bins_in_window = int(np.ceil(window_length / delta))
    return last_event_index, bins, bins_in_window


@dataclass
class PipelineResult:
    regression_results: List[PointProcessResult]
    taus: List[float]


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

    Info:
        This function implements the pipeline suggested by Riccardo Barbieri, Eric C. Matten, Abdul Rasheed A. Alabi,
        and Emery N. Brown in the paper:
        "A point-process model of human heartbeat intervals: new definitions of heart rate and heart rate variability"
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
    n_optimization_events = len(event_times) - len(observed_events)
    passed_events = 0
    # Initialize model parameters to None
    thetap = None
    k = None
    # Initialize result lists
    all_results: List[PointProcessResult] = []
    # _lambdas is just a temporary list containing the instantaneous hazard rate value, this list will be reset
    # each time we encounter a new event. This values are needed for the computation of the taus. (see below)
    _lambdas: List[float] = []
    # taus will contain the values needed to assess goodness of fit, refer to the following paper for more details:
    # "The time-rescaling theorem and its application to neural spike train data analysis"
    # (Emery N Brown, Riccardo Barbieri, Valérie Ventura, Robert E Kass, Loren M Frank)
    taus = []
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
            passed_events += 1
            last_event_index += 1
            # Append current event
            observed_events = np.append(observed_events, events[last_event_index])
            # Force re-evaluation of starting point for thetap
            thetap = None
            # Compute the tau value for the last observed event by approximating the integral:
            tau = np.sum(_lambdas) * delta
            taus.append(tau)
            # reset _lambdas
            _lambdas = []

        # Let's save the target event for the current time bin
        if last_event_index < len(events) - 1:
            target = events[last_event_index + 1] - events[last_event_index]
        else:
            # We can't know the target event time for the last event observed in our full dataset.
            target = None
        # Now if thetap is empty (i.e., observed_events has changed), re-evaluate the
        # variables that depend on observed_events
        if thetap is None:
            dataset = PointProcessDataset.load(
                event_times=observed_events,
                p=ar_order,
                hasTheta0=hasTheta0,
                current_time=current_time,
                target=target,
            )
            # The uncensored log-likelihood is a good starting point
            result = regr_likel(dataset, InterEventDistribution.INVERSE_GAUSSIAN)
            thetap, k = result.theta, result.k
        else:
            # If we end up in this branch, then the only thing that change is the right-censoring part,
            # we can use the thetap and k computed in the previous iteration as starting point for the optimization
            # process.
            pass

        wt = current_time - observed_events[-1]
        # 1e-2 is completely arbitrary, why did I put this? optimize time!
        if igcdf(wt, result.mu, result.k) < 1e-22:  # pragma: no cover

            # Copy the prior result
            result = deepcopy(result)
            # The only things that change are current time and lambda
            result.current_time = current_time

            result.lambda_ = compute_lambda(result.mu, result.k, wt)
        else:  # pragma: no cover
            # Let's optimize with right-censoring enabled
            dataset = PointProcessDataset.load(
                event_times=observed_events,
                p=ar_order,
                hasTheta0=hasTheta0,
                right_censoring=True,
                current_time=current_time,
                target=target,
            )
            result = regr_likel(
                dataset=dataset,
                maximizer_distribution=InterEventDistribution.INVERSE_GAUSSIAN,
                theta0=thetap.reshape(-1, 1),
                k0=k,
            )

        all_results.append(result)

        # FIXME since we are calculating the taus inside the main loop of the Regression Pipeline,
        #  We could avoid saving the lambda_ value inside the PointProcessResult and just compute it here
        _lambdas.append(result.lambda_)

        print(
            "\U0001F92F Currently evaluating time bin {:.3f} / {}  ({:.2f}%) \U0001F92F"
            "".format(
                current_time,
                (bins + 1) * delta,
                passed_events / n_optimization_events * 100,
            ),
            end="\r",
        )

    return PipelineResult(all_results, taus)
