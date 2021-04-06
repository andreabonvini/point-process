from typing import Union  # pragma: no cover

import matplotlib.pyplot as plt  # pragma: no cover
import numpy as np  # pragma: no cover


def compute_taus(
    lambdas: np.ndarray, events: Union[np.ndarray, list], delta: float
) -> np.ndarray:  # pragma: no cover
    """
    Args:
        lambdas (np.array): intantaneouus hazard-rate function for each time bini
        events (np.array): boolean array containing if an event occurred or not in a time bin
        delta (float): distance in time between two time bins.

    Returns:
        The approximated integral of the hazard-rate function lambda for each event happened.
        We'll have taus.shape = (n,) where n = np.sum(events) (number of events)
    """
    # In order to fully understand the meaning of taus refer to the following paper:
    # "The time-rescaling theorem and its application to neural spike train data analysis"
    # (Emery N Brown, Riccardo Barbieri, Valérie Ventura, Robert E Kass, Loren M Frank)
    taus = []
    int_lambda = 0
    for event, lamb in zip(events, lambdas):
        if event:
            taus.append(int_lambda)
            int_lambda = 0
        int_lambda += lamb * delta
    return np.array(taus)


def ks_distance(taus: np.ndarray, plot: bool = False):  # pragma: no cover
    """
    Compute KS-distance through the Time-Rescaling theorem
    """
    z = 1 - np.exp(-taus)
    z = sorted(z)
    d = len(z)
    lin = np.linspace(0, 1, d)
    if plot:
        plt.figure(figsize=(12, 8))
        lu = np.linspace(1.36 / np.sqrt(d), 1 + 1.36 / np.sqrt(d), d)
        ll = np.linspace(-1.36 / np.sqrt(d), 1 - 1.36 / np.sqrt(d), d)
        plt.plot(z, lin)
        plt.plot(lin, lin)
        plt.plot(lu, lin)
        plt.plot(ll, lin)
    KSdistance = max(abs(z - lin)) / np.sqrt(2)
    return KSdistance
