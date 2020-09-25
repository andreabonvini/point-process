from typing import Union

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


def plot_intervals(r: np.array) -> None:
    """
    data: np.array as returned from the load() function.

    This function just plot the time (t) and RR intervals (rr) respectively on the x and y axis.
    """

    diff = np.diff(r)
    plt.xlabel("time [s]")
    plt.ylabel("RR [ms]")
    plt.plot(r[:-1], diff)


def unpack_invgauss_params(params: np.array, m: int, n: int):
    return params[0], params[1 : 1 + m].reshape((m, 1)), params[1 + m :].reshape((n, 1))


def likel_invgauss_consistency_check(
    xn: np.array,
    wn: np.array,
    eta: Union[np.array, None],
    xt: Union[np.array, None],
    thetap0: Union[np.array, None],
):
    m, n = xn.shape
    if wn.shape != (m, 1):
        raise ValueError(
            f"Since xn has shape {xn.shape}, wn should be of shape ({m},1).\n"
            f"Instead wn has shape {wn.shape}"
        )
    if eta is not None and eta.shape != (m, 1):
        raise ValueError(
            f"Since xn has shape {xn.shape}, eta should be of shape ({m},1).\n"
            f"Instead eta has shape {eta.shape}"
        )
    if xt is not None and xt.shape != (1, n):
        raise ValueError(
            f"Since xn has shape {xn.shape}, xt should be of shape (1,{n}).\n"
            f"Instead xt has shape {xt.shape}"
        )
    if thetap0 is not None and thetap0.shape != (n, 1):
        raise ValueError(
            f"Since xn has shape {xn.shape}, thetap0 should be of shape ({n},1).\n"
            f"Instead thetap0 has shape {thetap0.shape}"
        )
