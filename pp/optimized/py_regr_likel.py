import ctypes
import os

import numpy as np


def regr_likel(
    AR_ORDER: int,
    N_SAMPLES: int,
    max_steps: int,
    theta0: np.ndarray,
    k0: float,
    xn: np.ndarray,
    eta: np.ndarray,
    wn: np.ndarray,
    xt: np.ndarray,
    wt: float,
) -> np.ndarray:
    # Convert np.array to ctype doubles
    theta0 = theta0.astype(np.double)
    theta0_p = theta0.ctypes.data_as(c_double_p)
    xn = xn.astype(np.double)
    xn_p = xn.ctypes.data_as(c_double_p)
    eta = eta.astype(np.double)
    eta_p = eta.ctypes.data_as(c_double_p)
    wn = wn.astype(np.double)
    wn_p = wn.ctypes.data_as(c_double_p)
    xt = xt.astype(np.double)
    xt_p = xt.ctypes.data_as(c_double_p)

    regr_likel_cdll.regr_likel.restype = np.ctypeslib.ndpointer(
        dtype=ctypes.c_double, shape=(AR_ORDER + 2,)
    )
    # Compute result...
    return regr_likel_cdll.regr_likel(
        AR_ORDER, N_SAMPLES, max_steps, theta0_p, k0, xn_p, eta_p, wn_p, xt_p, wt
    )


c_double_p = ctypes.POINTER(ctypes.c_double)
# so_file generated with:
# cc -framework Accelerate -lnlopt -fPIC -shared -o regr_likel.so c_regr_likel.c
# "-framework Accelerate -lnlopt" -> Linking Accelerate.h and nlopt
so_file = "/Users/z051m4/PycharmProjects/pointprocess/pp/optimized/regr_likel.so"
if not os.path.isfile(so_file):  # pragma: no cover
    raise Exception(
        "You have to generate a .so file first.\nCheck the README.md file associated with this repository."
    )
else:
    regr_likel_cdll = ctypes.CDLL(so_file)
    regr_likel_cdll.regr_likel.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        c_double_p,
        ctypes.c_double,
        c_double_p,
        c_double_p,
        c_double_p,
        c_double_p,
        ctypes.c_double,
    ]
