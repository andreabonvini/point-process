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
        expected = np.array([1.0, 1.0, 1.0, 1.0])
        res = ConstantWeightsProducer()(len(self.wn))
        self.assertEqual(res.all(), expected.all())

    def test_exponential_weighs(self):
        expected = np.array([1.0, 0.60653066, 0.36787944, 0.22313016])
        res = ExponentialWeightsProducer()(target_intervals=self.wn, alpha=1.0)
        self.assertEqual(res.all(), expected.all())
