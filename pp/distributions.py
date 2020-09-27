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
    if isinstance(xs, np.ndarray) and isinstance(mus, np.ndarray):
        if xs.shape != mus.shape:
            raise ValueError(
                f"{xs.shape}!={mus.shape}.\n"
                "xs and mus should have the same shape if they're both np.array"
            )
    elif isinstance(xs, np.ndarray) or isinstance(mus, np.ndarray):
        raise TypeError(
            f"xs: {type(xs)}\n"
            f"mus: {type(mus)}\n"
            f"xs and mus should be either both np.array or both float"
        )
    arg = lamb / (2 * np.pi * xs ** 3)
    return np.sqrt(arg) * np.exp((-lamb * (xs - mus) ** 2) / (2 * mus ** 2 * xs))
