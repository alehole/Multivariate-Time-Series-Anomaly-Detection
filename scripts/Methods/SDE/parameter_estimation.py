import numpy as np
from SDE_config import THETA0, LOWER_BOUND, UPPER_BOUND, Q, R, C,PARAMETER_NAMES

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

def estimate_parameters_mle(df_train):
    theta0 = np.asarray(THETA0, float)
    lo = np.asarray(LOWER_BOUND, float)
    hi = np.asarray(UPPER_BOUND, float)

    m0 = np.ones_like(theta0)
    lo_scaled = lo / theta0
    hi_scaled = hi / theta0

    def nll_scaled(m):
        theta = m * theta0
        return neg_log_likelihood(
            theta,
            df_train,
            C,
            Q,
            R,
        )
    initial_nll = nll_scaled(m0)
    result = minimize_nll(
        nll_scaled,
        m0,
        lo_scaled,
        hi_scaled,
    )
    theta_hat = result.x * theta0
    final_nll = nll_scaled(result.x)
    print("Optimization success:", result.success)
    print("Optimizer message:", result.message)
    print("Initial parameters:", theta0)
    print("Estimated parameters:", theta_hat)
    print("Parameter bounds (lo, hi):", (lo, hi))
    print("Initial NLL:", initial_nll)
    print("Final NLL:", final_nll)
    print("NLL reduction:", initial_nll - final_nll)

    tol = 1e-3
    at_lo = np.abs(theta_hat - lo) / np.abs(lo) < tol
    at_hi = np.abs(theta_hat - hi) / np.abs(hi) < tol
    if at_lo.any() or at_hi.any():
        stuck = [n for n, f in zip(PARAMETER_NAMES, at_lo | at_hi) if f]
        print(f"WARNING: parameters pinned at bounds: {stuck} -- widen and re-run")


    return theta_hat, result

from scipy.stats import chi2

def wilks_likelihood_ratio_test(nll_reduced, nll_larger, num_extra_theta):
    """
     Perform Wilks' likelihood-ratio test for two nested models.
     The two models must be nested — the reduced one has to be a special case of the full one

    nll_reduced         :  Negative log-likelihood of the reduced model evaluated at its MLE.
    nll_larger          : Negative log-likelihood of the full model evaluated at its MLE.
    num_extra_theta     : Difference in the number of freely estimated parameters:


    returns
        test_statistic: Likelihood-ratio test statistic.
         p_value: Upper-tail probability under a chi-squared distribution.
    """
    LR = nll_reduced - nll_larger        # = (lnL_full − nll_larger) = the Eq.(21) Brastein paper
    p_value = chi2.sf(LR, num_extra_theta)
    return LR, p_value