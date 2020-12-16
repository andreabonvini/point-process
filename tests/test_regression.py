from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from pp.core.maximizers import InverseGaussianMaximizer
from pp.model import InterEventDistribution, PointProcessDataset, PointProcessResult
from pp.regression import _pipeline_setup, regr_likel, regr_likel_pipeline

mock_maximizer = Mock(spec=InverseGaussianMaximizer)
mock_maximizer.train.return_value = PointProcessResult(
    True, np.array([[1.0], [2.0], [3.0]]), 500.0, 5.0, 1.0, 1.0, 1.0, 1.0, 0.8
)


def fake_constructor(dataset, theta0, k0, verbose, save_history):
    return mock_maximizer


mocked_maximizers_dict = {
    InterEventDistribution.INVERSE_GAUSSIAN.value: fake_constructor
}


class TestRegression(TestCase):
    @patch.dict("pp.regression.maximizers_dict", mocked_maximizers_dict)
    def test_regr_likel(self):
        events = np.linspace(1, 30, 15)
        res = regr_likel(
            dataset=PointProcessDataset.load(events, 3, True),
            maximizer_distribution=InterEventDistribution.INVERSE_GAUSSIAN,
        )
        self.assertIsInstance(res, PointProcessResult)

    def test_pipeline_setup_1(self):
        events = np.array([0.0 + 0.9 * i for i in range(20)])
        # events: [0., 0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1, 9.,
        # 9.9, 10.8, 11.7, 12.6, 13.5, 14.4, 15.3, 16.2, 17.1]
        # First event at 0.0 s, last event a 17.1 seconds
        window_length = (
            10  # We perform regression each time shifting a 10 seconds window
        )
        time_resolution = 0.5  # We update the parameters every 0.5 seconds
        # (i.e. we shift the window defined above each time by 0.5 seconds)
        last_event_index, bins, bins_in_window = _pipeline_setup(
            events, window_length, time_resolution
        )
        self.assertEqual(last_event_index, 11)  # -> the sample at index 11 is 9.9, i.e.
        # the last event time before the window_length (10. s)
        self.assertEqual(bins, 35)  # 17.1 can be divided in 35 bins of width 0.5
        self.assertEqual(bins_in_window, 20)

    def test_pipeline_setup_2(self):
        events = np.array([0.0, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1, 4.6])
        # First event at 0.0 s, last event a 4.6 seconds
        window_length = 3  # We perform regression each time shifting a 3 seconds window
        delta = 0.5  # We update the parameters every 0.5 seconds
        # (i.e. we shift the window defined above each time by 0.5 seconds)
        last_event_index, bins, bins_in_window = _pipeline_setup(
            events, window_length, delta
        )
        self.assertEqual(last_event_index, 5)  # -> the sample at index 5 is 2.6, i.e.
        # the last event time before the window_length (3. s)
        self.assertEqual(bins, 10)  # 4.6 can be divided in 10 bins of width 0.5
        self.assertEqual(bins_in_window, 6)

    def test_pipeline_setup_3(self):
        events = np.array([0.0, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1, 4.6])
        # First event at 0.0 s, last event a 4.6 seconds
        window_length = 3  # We perform regression each time shifting a 3 seconds window
        delta = 0.4  # We update the parameters every 0.4 seconds
        # (i.e. we shift the window defined above each time by 0.4 seconds)
        last_event_index, bins, bins_in_window = _pipeline_setup(
            events, window_length, delta
        )
        self.assertEqual(last_event_index, 5)  # -> the sample at index 5 is 2.6, i.e.
        # the last event time before the window_length (3. s)
        self.assertEqual(bins, 12)  # 4.6 can be divided in 12 bins of width 0.4
        self.assertEqual(
            bins_in_window, 8
        )  # In order fully observe 3.0 seconds we need 8 bins (3.2 s)

    def test_pipeline_setup_invalid(self):
        events = np.array([0, 1.0, 2.0, 3.0, 4.0])
        window_length = (
            10  # We perform regression each time shifting a 10 seconds window
        )
        delta = 0.5
        with self.assertRaises(Exception):
            _pipeline_setup(events, window_length, delta)

    @patch.dict("pp.regression.maximizers_dict", mocked_maximizers_dict)
    def test_regr_likel_pipeline(self):
        events = np.array([0.0, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 3.6, 4.1, 4.6])
        pip_result = regr_likel_pipeline(
            event_times=events, ar_order=3, hasTheta0=True, window_length=3.0, delta=0.5
        )
        self.assertIsInstance(pip_result.regression_results[0], PointProcessResult)
