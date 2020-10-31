from unittest import TestCase

import numpy as np

from pp.core.distributions import events2interevents


class TestUtils(TestCase):
    def setUp(self) -> None:

        self.events = np.array([0.000, 0.500, 0.900, 1.500, 2.200, 3.200, 4.100])
        self.inter_event_times = np.array([500.0, 400.0, 600.0, 700.0, 1000.0, 900.0])

    def test_events2interevents(self):
        res = events2interevents(self.events)
        self.assertEqual(res.all(), self.inter_event_times.all())
