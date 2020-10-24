from enum import Enum
from typing import Callable

import numpy as np


class InterEventDistribution(Enum):
    INVERSE_GAUSSIAN = "Inverse Gaussian"


class PointProcessResult:
    def __init__(self, mu, sigma):  # pragma: no cover
        self.mu = mu
        self.sigma = sigma

    def __repr__(self):
        return f"mu: {self.mu}\nsigma: {self.sigma}"


class PointProcessModel:
    def __init__(
        self,
        model: Callable[[np.ndarray], PointProcessResult],
        distribution: InterEventDistribution,
        ar_order: int,
    ):
        self._model = model
        self.distribution = distribution
        self.ar_order = ar_order

    def __repr__(self):
        return (
            f"<PointProcessModel <<model={self._model}>"
            f"<distributuon={self.distribution}>"
            f"<ar_order={self.ar_order}>>"
        )

    def __call__(self, *args, **kwargs):  # pragma: no cover
        return self._model(*args)
