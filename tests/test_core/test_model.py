from unittest import TestCase

import numpy as np

from data import InverseGaussianTestData
from pp.core.model import PointProcessDataset


class TestModel(TestCase):
    def setUp(self) -> None:
        self.data = InverseGaussianTestData()

    def test_dataset_load(self):
        dataset = PointProcessDataset.load(self.data.inter_events_times, self.data.p)
        np.testing.assert_array_equal(dataset.xn, self.data.xn)
        np.testing.assert_array_equal(dataset.wn, self.data.wn)
