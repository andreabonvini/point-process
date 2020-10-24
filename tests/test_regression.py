from unittest import TestCase
from unittest.mock import patch

import numpy as np

from pp.core.model import InterEventDistribution, PointProcessModel, PointProcessResult
from pp.regression import regr_likel


def fake_model(events: np.ndarray) -> PointProcessResult:
    return PointProcessResult(1.0, 1.0)


mock_res = PointProcessModel(
    model=fake_model, distribution=InterEventDistribution.INVERSE_GAUSSIAN, ar_order=3
)

mocked_maximizers_dict = {
    InterEventDistribution.INVERSE_GAUSSIAN.value: lambda xn, wn: mock_res
}


class TestPointProcess(TestCase):
    @patch.dict("pp.regression.maximizers_dict", mocked_maximizers_dict)
    def test_regr_likel(self):
        events = np.linspace(1, 30, 15)
        res = regr_likel(
            events=events, maximizer=InterEventDistribution.INVERSE_GAUSSIAN, p=3
        )
        self.assertEqual(res, mock_res)
