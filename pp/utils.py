import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load(path: str) -> np.array:
    """
    path:   Path to the .csv file containing the raw data to process. See test_data/Y2.csv for an example

    The first column in Y2.csv is the elapsed time since the beginning of the recording (in seconds), and the second
    column is the instantaneous heart rate (in beats/minute).
    """
    data = np.array(pd.read_csv(path))
    r = data[:, 0]
    # for now we ignore data[:, 1] since #TODO
    return r


def plot(r: np.array) -> None:
    """
    data: np.array as returned from the load() function.

    This function just plot the time (t) and RR intervals (rr) respectively on the x and y axis.
    """

    diff = np.diff(r)
    plt.xlabel("time [s]")
    plt.ylabel("RR [ms]")
    plt.plot(r, diff)
