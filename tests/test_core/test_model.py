from unittest import TestCase

import numpy as np

from pp.model import PointProcessDataset
from tests.data import DatasetTestData


class TestModel(TestCase):
    def setUp(self) -> None:
        self.data = DatasetTestData()

    def test_dataset_load(self):
        dataset = PointProcessDataset.load(self.data.events, self.data.p)
        np.testing.assert_array_equal(dataset.xn, self.data.xn)
        np.testing.assert_array_equal(dataset.wn, self.data.wn)

    def test_dataset_wt_wrong(self):
        xn = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        wn = np.array([[1.0], [1.0], [1.0]])
        p = 2
        hasTheta = True
        eta = np.array([[1.0, 1.0, 1.0]])
        current_time = 5.0
        xt = np.array([[1.0, 1.0, 1.0]])
        target = 1.0
        wt = -0.5
        with self.assertRaises(AssertionError):
            PointProcessDataset(xn, wn, p, hasTheta, eta, current_time, xt, target, wt)
