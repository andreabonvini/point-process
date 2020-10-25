from unittest import TestCase

import numpy as np

from pp.core.model import PointProcessDataset


class TestModel(TestCase):
    def setUp(self) -> None:
        self.p = 3
        self.inter_events_times = np.array([500.0, 400.0, 600.0, 700.0, 1000.0, 900.0])
        self.xn = np.array(
            [
                [1.00, 600.0, 400.0, 500.0],
                [1.00, 700.0, 600.0, 400.0],
                [1.00, 1000.0, 700.0, 600.0],
            ]
        )
        self.wn = np.array([[900.0, 1000.0, 700.0]])

    def test_dataset_load(self):
        dataset = PointProcessDataset.load(self.inter_events_times, self.p)
        self.assertEqual(dataset.xn.all(), self.xn.all())
        self.assertEqual(dataset.wn.all(), self.wn.all())
