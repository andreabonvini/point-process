from unittest import TestCase

import numpy as np

from pp.core.weights_producers import ExponentialWeightsProducer


class TestWeightsProducers(TestCase):
    def setUp(self) -> None:
        self.target_distances = np.array([2.0, 1.5, 1.0, 0.5, 0.0])

    def test_exponential_weighs(self):
        expected = np.array([[0.81], [0.85381497], [0.9], [0.9486833], [1.0]])
        res = ExponentialWeightsProducer(alpha=0.9)(
            target_distances=self.target_distances
        )
        np.testing.assert_array_almost_equal(res, expected, decimal=2)
