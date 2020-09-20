from typing import Union

import numpy as np


def inverse_gaussian(
    xs: Union[np.array, float], mus: Union[np.array, float], lamb: float
):
    """
    @param xs: points or point in which evaluate the probabilty
    @type xs: np.array or float
    @param mus: inverse gaussian means or mean
    @type mus: np.array or float
    @param lamb: inverse gaussian scaling factor
    @type lamb: float
    @return: p: probability values, 0 < p < 1
    @rtype: np.array
    """
    if isinstance(xs, np.array) and isinstance(mus, np.array):
        if xs.shape != mus.shape:
            raise TypeError(
                f"{xs.shape}!={mus.shape}.\n"
                "xs and mus should have the same shape if they're both np.array"
            )
    elif isinstance(xs, np.array) or isinstance(mus, np.array):
        raise TypeError(
            f"xs: {type(xs)}\n"
            f"mus: {type(mus)}\n"
            f"xs and mus should be either both np.array or both float"
        )
    return np.sqrt(lamb / (2 * np.pi * xs ** 3)) * np.exp(
        (-lamb * (xs - mus) ** 2) / (2 * mus ** 2 * xs)
    )
