from typing import List

import numpy as np
from scipy.linalg import toeplitz

from pp.likelihood import likel_invnorm


def regr_likel(events: np.array, opt: dict = None) -> List:
    """

    @param events: event-times as returned by the pp.utils.load() function
    @type events: np.array
    @param opt: dictionary containing the main options for the regression procedure
    (see definition of opt in the code below for more details).
    @type opt: dict
    @return: [thetap, kappa, opt]
    The variable thetap is a vector with the weight of the previous RR intervals in the regression of the first moment
    (µ) of the IG while kappa is the shape parameter of the IG (it is often called λ but we use Kappa to avoid confusion
    with the lambda function that we will introduce later).
    @rtype: list

    Copyright(C) - Luca Citi and Riccardo Barbieri, 2010 - 2011.
    All Rights Reserved.
    See LICENSE.TXT for license details.
    {lciti, barbieri} @ neurostat.mit.edu
    http://users.neurostat.mit.edu/barbieri/pphrv
    """

    if opt is None:
        opt = {
            "P": 9,  # RR order
            "hasTheta0": True,  # whether or not the AR model has a theta0 constant to account for the average mu
            "maximize_loglikel": likel_invnorm,  # use loglikelihood of inverse gaussian
        }

    # We reset the events s.t. the first event is at time 0.
    observ_ev = events - events[0]

    # TODO uk is probably useless.
    # uk = observ_ev[opt["P"] + 1 :]

    # rr is a np.array which contains the inter-event intervals.
    rr = np.diff(observ_ev)

    # wn are the target inter-event intervals, i.e. the intervals we have to predict once we build our
    # RR autoregressive model.
    wn = rr[opt["P"] :]

    # We now have to build a matrix xn s.t. for i = 0, ..., len(rr)-p-1 the i_th element of xn will be
    # xn[i] = [1, rr[i + p - 1], rr[i + p - 2], ..., rr[i]]
    # Note that the 1 at the beginning of each row is added only if the hasTheta0 parameter is set to True.
    a = rr[opt["P"] - 1 : -1]
    b = rr[opt["P"] - 1 :: -1]
    xn = toeplitz(a, b)
    if opt["hasTheta0"]:
        xn = np.hstack([np.ones((len(wn), 1)), xn])

    [thetap, kappa, steps, loglikel] = opt["maximize_loglikel"](xn, wn)

    opt["steps"] = steps
    opt["loglikel"] = loglikel

    return [thetap, kappa, opt]
