import numpy as np


class ExponentialWeightsProducer:
    def __init__(self, alpha: float = 0.98):
        """
        Args:
            alpha: Weighting time constant that governs the degree of influence
                    of a previous observation on the local likelihood.
        """
        self.alpha = alpha

    def __call__(self, target_distances: np.ndarray) -> np.ndarray:
        self.target_distances = target_distances
        return self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        return np.exp(np.log(self.alpha) * self.target_distances).reshape(-1, 1)
