from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from pp.core.model import (
    InterEventDistribution,
    PointProcessMaximizer,
    PointProcessModel,
    PointProcessResult,
)
from pp.regression import regr_likel


def fake_model(events: np.ndarray) -> PointProcessResult:
    return PointProcessResult(1.0, 1.0)


mock_model = PointProcessModel(
    model=fake_model,
    expected_shape=(3,),
    distribution=InterEventDistribution.INVERSE_GAUSSIAN,
    ar_order=3,
    hasTheta0=True,
)

mock_maximizer = Mock(spec=PointProcessMaximizer)
mock_maximizer.train.return_value = mock_model

mocked_maximizers_dict = {
    InterEventDistribution.INVERSE_GAUSSIAN.value: lambda data: mock_maximizer
}


class TestPointProcess(TestCase):
    @patch.dict("pp.regression.maximizers_dict", mocked_maximizers_dict)
    def test_regr_likel(self):
        events = np.linspace(1, 30, 15)
        res = regr_likel(
            events=events,
            maximizer_distribution=InterEventDistribution.INVERSE_GAUSSIAN,
            p=3,
        )
        self.assertEqual(res, mock_model)
