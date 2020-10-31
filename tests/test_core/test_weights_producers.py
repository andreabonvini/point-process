from unittest import TestCase

import numpy as np

from pp.core.weights_producers import (
    ConstantWeightsProducer,
    ExponentialWeightsProducer,
)


class TestWeightsProducers(TestCase):
    def setUp(self) -> None:
        self.wn = np.array([0.5, 0.5, 0.5, 0.5])

    def test_constant_weighs(self):
        expected = np.array([[1.0], [1.0], [1.0], [1.0]])
        res = ConstantWeightsProducer()(len(self.wn))
        np.testing.assert_array_equal(res, expected)

    def test_exponential_weighs(self):
        expected = np.array([[0.22313016], [0.36787944], [0.60653066], [1.0]])
        res = ExponentialWeightsProducer(alpha=1.0)(target_intervals=self.wn)
        np.testing.assert_array_almost_equal(res, expected)
