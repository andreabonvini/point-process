import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(path: str) -> np.array:
    """
    path:   Path to the .csv file containing the raw data to process. See test_data/Y2.csv for an example

    The first column in Y2.csv is the elapsed time since the beginning of the recording (in seconds), and the second
    column is the instantaneous heart rate (in beats/minute).
    # FIXME This should be a generic function which returns a np.array with the even times expressed in seconds.
    """
    data = np.array(pd.read_csv(path))
    r = data[:, 0]
    # for now we ignore data[:, 1] since #TODO
    return r


def events2interevents(events: np.array) -> np.array:
    """
    Args:
        events: event-times as returned by the pp.utils.load() function.
                MUST be expressed in <seconds>.

    Returns:
         np.array which contains the inter-event intervals expressed in ms.
    """
    # We reset the events s.t. the first event is at time 0.
    # FIXME actually useless
    observ_ev = events - events[0]

    return np.diff(observ_ev) * 1000


def plot_intervals(r: np.array) -> None:
    """
    data: np.array as returned from the load() function.

    This function just plot the time (t) and RR intervals (rr) respectively on the x and y axis.
    """

    diff = np.diff(r)
    plt.xlabel("time [s]")
    plt.ylabel("RR [ms]")
    plt.plot(r[:-1], diff)
