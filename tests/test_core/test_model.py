from unittest import TestCase

import numpy as np

from pp.core.model import PointProcessDataset
from tests import DatasetTestData


class TestModel(TestCase):
    def setUp(self) -> None:
        self.data = DatasetTestData()

    def test_dataset_load(self):
        dataset = PointProcessDataset.load(self.data.inter_events_times, self.data.p)
        np.testing.assert_array_equal(dataset.xn, self.data.xn)
        np.testing.assert_array_equal(dataset.wn, self.data.wn)
