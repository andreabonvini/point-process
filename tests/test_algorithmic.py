from unittest.mock import Mock, patch

import numpy as np
from scipy.optimize.optimize import OptimizeResult

from pp.algorithmic import regr_likel


@patch("pp.algorithmic.likel_invnorm")
def test_regr_likel(li):
    mock_res = Mock(OptimizeResult)
    li.return_value = mock_res
    events = np.linspace(1, 30, 15)
    res = regr_likel(events)
    assert res == mock_res
