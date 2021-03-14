from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from pp.core.distributions.inverse_gaussian import igcdf, igpdf
from pp.core.maximizers import InverseGaussianMaximizer
from pp.model import InterEventDistribution, InverseGaussianResult, PointProcessDataset

maximizers_dict = {
    InterEventDistribution.INVERSE_GAUSSIAN.value: InverseGaussianMaximizer
}


def compute_lambda_approximation(
    mu: float, k: float
) -> Callable[[float], float]:  # pragma: no cover
    wts = np.linspace(0.25, 1.6, 1000)
    old = -1
    y = []
    x = []
    for i in range(len(wts)):
        current_lambda = igpdf(wts[i], mu, k) / (1 - igcdf(wts[i], mu, k))
        if current_lambda >= old:
            if current_lambda > 10:
                y.append(current_lambda)
                x.append(wts[i])
            old = current_lambda
        else:
            break
    X = np.array(x).reshape((-1, 1))
    model = LinearRegression().fit(X, y)

    def lambda_approximation(t: float) -> float:
        return model.intercept_ + model.coef_[0] * t

    return lambda_approximation


def regr_likel(
    dataset: PointProcessDataset,
    maximizer_distribution: InterEventDistribution,
    theta0: Optional[np.ndarray] = None,
    k0: Optional[float] = None,
    max_steps: int = 50,
) -> InverseGaussianResult:
    """
    Args:
        dataset: PointProcessDataset object
        maximizer_distribution: log-likelihood maximization function belonging to the Maximizer enum.
        theta0: starting vector for the theta parameters
        k0: starting value for the k parameter
        max_steps: Maximum number of iterations during the optimization procedure

    Returns:
        PointProcessResult
    """

    return maximizers_dict[maximizer_distribution.value](
        dataset=dataset, max_steps=max_steps, theta0=theta0, k0=k0,
    ).train()


def _pipeline_setup(
    event_times: np.ndarray, window_length: float, delta: float
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
    regression_results: List[InverseGaussianResult]
    taus: List[float]


def regr_likel_pipeline(
    event_times: np.ndarray,
    ar_order: int = 9,
    window_length: float = 60.0,
    delta: float = 0.005,
) -> PipelineResult:  # pragma: no cover
    """
    Args:
        event_times: event times expressed in seconds.
        ar_order: AR order to use in the regression process
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
    # Initialize model parameters to None
    thetap = None
    k = None
    # Initialize result lists
    all_results: List[InverseGaussianResult] = []
    # lambdas is just a temporary list containing the instantaneous hazard rate value, this list will be reset
    # each time we encounter a new event. This values are needed for the computation of the taus. (see below)
    lambdas: List[float] = []
    # taus will contain the values needed to assess goodness of fit, refer to the following paper for more details:
    # "The time-rescaling theorem and its application to neural spike train data analysis"
    # (Emery N Brown, Riccardo Barbieri, Valérie Ventura, Robert E Kass, Loren M Frank)
    taus = []
    # When we'll have to compute the Hazard Function for t > mu it may happen that we encounter some numerical problem,
    #  in this case we'll use an approximation (specifically we'll estimate an approximation by means of a
    #  simple linear regression)
    lambda_approximation: Callable[[float], float]
    lambda_stable = True
    old_lambda = -1
    for bin_index in tqdm(
        range(bins_in_window, bins + 1)
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
            # Compute the tau value for the last observed event by approximating the integral:
            tau = sum(lambdas) * delta
            taus.append(tau)
            # reset _lambdas
            lambdas = []
            old_lambda = -1
            lambda_stable = True

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
                current_time=current_time,
                target=target,
            )
            # The uncensored log-likelihood is a good starting point
            result = regr_likel(
                dataset, InterEventDistribution.INVERSE_GAUSSIAN, max_steps=50
            )
            thetap, k = result.theta, result.k
        else:
            # If we end up in this branch, then the only thing that change is the right-censoring part,
            # we can use the thetap and k computed in the previous iteration as starting point for the optimization
            # process.
            pass

        dataset = PointProcessDataset.load(
            event_times=observed_events,
            p=ar_order,
            current_time=current_time,
            target=target,
        )
        wt = current_time - observed_events[-1]
        mu = np.dot(dataset.xt, thetap.reshape(-1, 1))[0, 0]
        cdf = igcdf(wt, mu, k)
        pdf = igpdf(wt, mu, k)
        # current_lambda: Inhomogenous Poisson rate (or hazard function) at current_time
        if lambda_stable:
            current_lambda = pdf / (1 - cdf)
        else:
            current_lambda = lambda_approximation(wt)  # noqa
            assert current_lambda > 0.0
        if old_lambda > current_lambda > 0.1 and lambda_stable:
            lambda_stable = False
            lambda_approximation = compute_lambda_approximation(mu, k)
            current_lambda = lambda_approximation(wt)
            assert current_lambda > 0.0
        old_lambda = current_lambda
        lambdas.append(current_lambda)

        if cdf > 0.0:
            # Right-censoring is induced only when the cdf is > 0.0
            result = regr_likel(
                dataset=dataset,
                maximizer_distribution=InterEventDistribution.INVERSE_GAUSSIAN,
                theta0=thetap.reshape(-1, 1),
                k0=k,
                max_steps=10,
            )
        else:
            result = deepcopy(result)
            result.current_time = current_time

        all_results.append(result)

    return PipelineResult(all_results, taus)
