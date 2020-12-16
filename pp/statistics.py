from typing import List

import matplotlib.pyplot as plt
import numpy as np


def ks_distance(taus: List[float], plot: bool = False):
    """
    Compute KS-distance through the Time-Rescaling theorem
    """
    z = 1 - np.exp(-np.array(taus))
    z = sorted(z)
    d = len(z)
    lin = np.linspace(0, 1, d)
    if plot:  # pragma: no cover
        lu = np.linspace(1.36 / np.sqrt(d), 1 + 1.36 / np.sqrt(d), d)
        ll = np.linspace(-1.36 / np.sqrt(d), 1 - 1.36 / np.sqrt(d), d)
        plt.plot(z, lin)
        plt.plot(lin, lin)
        plt.plot(lu, lin)
        plt.plot(ll, lin)
    KSdistance = max(abs(z - lin)) / np.sqrt(2)
    return KSdistance
