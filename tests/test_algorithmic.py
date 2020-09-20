from unittest.mock import patch

import numpy as np

from pp.algorithmic import regr_likel


@patch("pp.algorithmic.likel_invnorm")
def test_regr_likel(li):
    fake_thetap = np.ones((10, 1))
    fake_kappa = 0.5
    fake_steps = 20
    fake_loglikel = 42.42
    fake_opt = {
        "P": 9,
        "hasTheta0": True,
        "maximize_loglikel": li,
        "steps": 20,
        "loglikel": 42.42,
    }
    li.return_value = [fake_thetap, fake_kappa, fake_steps, fake_loglikel]
    events = np.linspace(1, 30, 15)
    res = regr_likel(events)
    assert res == [fake_thetap, fake_kappa, fake_opt]
