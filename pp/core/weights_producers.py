import numpy as np

from pp.core.model import WeightsProducer


class ConstantWeightsProducer(WeightsProducer):
    def __call__(self, n: int) -> np.ndarray:
        self.n = n
        return self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        return np.ones((self.n,))


class ExponentialWeightsProducer(WeightsProducer):
    def __call__(self, target_intervals: np.ndarray, alpha: float) -> np.ndarray:
        """
            Args:
                target_intervals:
                    Target intervals vector (as stored in PointProcessDataset.wn)
                alpha:
                    Weighting time constant that governs the degree of influence
                    of a previous observation on the local likelihood.
        """
        self.target_intervals = target_intervals
        self.alpha = alpha
        return self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        target_times = np.cumsum(self.target_intervals) - self.target_intervals[0]
        return np.exp(-self.alpha * target_times)
