import numpy as np

from pp.core.model import WeightsProducer


class ConstantWeightsProducer(WeightsProducer):
    def __call__(self, n: int) -> np.ndarray:
        self.n = n
        return self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        return np.ones((self.n, 1))


class ExponentialWeightsProducer(WeightsProducer):
    def __init__(self, alpha: float = 0.005):
        """
        Args:
            alpha: Weighting time constant that governs the degree of influence
                    of a previous observation on the local likelihood.
        """
        self.alpha = alpha

    def __call__(self, target_intervals: np.ndarray) -> np.ndarray:
        """
            Args:
                target_intervals:
                    Target intervals vector (as stored in PointProcessDataset.wn)
        """
        self.target_intervals = target_intervals
        return self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        target_times = np.cumsum(self.target_intervals) - self.target_intervals[0]
        return np.exp(-self.alpha * target_times).reshape(-1, 1)[::-1]
