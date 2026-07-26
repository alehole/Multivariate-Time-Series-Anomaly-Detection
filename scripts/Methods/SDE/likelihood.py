import numpy as np
from scipy.optimize import minimize
from ekf import run_ekf
from SDE_config import C, Q, R, MAXITER

def neg_log_likelihood(theta, df, C=C, Q=Q, R=R):
    *_, nll = run_ekf(df, theta, C, Q, R)
    return nll

def minimize_nll(
    objective,
    x0,
    bound_lo,
    bound_hi,
    method="Powell",
    ):

    result = minimize(
        objective,
        np.asarray(x0, dtype=float),
        method=method,
        bounds=[(float(l), float(h)) for l, h in zip(bound_lo, bound_hi)],
        options={
            "maxiter": MAXITER ,
            "disp": True,
            "xtol": 1e-4,
            "ftol": 1e-4,
        },
    )
    return result