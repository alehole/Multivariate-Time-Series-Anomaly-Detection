import numpy as np
from scipy.stats import chi2
from scipy.optimize import minimize

from ekf import run_ekf
from SDE_config import(
    THETA0,
    LOWER_BOUND,
    UPPER_BOUND,
    PARAMETER_NAMES,
    C,
    R,
    Q_INIT,
    MAXITER,
)

def neg_log_likelihood(
        theta,
        df,
        C=C,
        Q=Q_INIT,
        R=R
):
    """
    Calculate the negative log-likelihood given the data, parameters C, Q, R.
    """
    *_, nll = run_ekf(df, theta, C, Q, R)
    return nll

def minimize_nll(
    objective,
    x0,
    bound_lo,
    bound_hi,
    method="Powell",
    ):
    """
    Minimize the negative log-likelihood objective subject to lower and upper parameter bounds.

    """
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
    """
    Estimate the thermal-model parameters by maximum likelihood.

    The optimization is performed using dimensionless scaling factors
    m, where theta = m * theta0. This improves numerical conditioning
    when the physical parameters have very different magnitudes.
    """
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
            Q_INIT,
            R,
        )
    initial_nll = nll_scaled(m0)

    # Find the scaled parameters that minimize the negative log-likelihood
    result = minimize_nll(
        nll_scaled,
        m0,
        lo_scaled,
        hi_scaled,
    )
    # Convert the optimized scaled parameters back to physical units.
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


def estimate_parameters_mle_q_theta(df_train):
    """
    Estimate the thermal-model parameters and diagonal process-noise
    covariance Q by maximum likelihood.

    The thermal parameters are optimized using dimensionless scaling
    factors m, where theta = m * theta0.

    The diagonal elements of Q are optimized in log-space to ensure
    positive process-noise variances.
    """

    theta0 = np.asarray(THETA0, float)
    lo = np.asarray(LOWER_BOUND, float)
    hi = np.asarray(UPPER_BOUND, float)

    # ---------------------------------------------------------
    # Thermal parameters
    # ---------------------------------------------------------
    m0 = np.ones_like(theta0)

    lo_scaled = lo / theta0
    hi_scaled = hi / theta0

    # ---------------------------------------------------------
    # Process-noise covariance Q
    # ---------------------------------------------------------
    Q0 = np.asarray(Q_INIT, float)

    # Initial diagonal process-noise variances
    q0 = np.diag(Q0)

    # Optimize Q in log-space so q > 0
    log_q0 = np.log(q0)

    # Bounds for Q
    q_lower = 1e-8
    q_upper = 1.0

    log_q_lo = np.full_like(log_q0, np.log(q_lower))
    log_q_hi = np.full_like(log_q0, np.log(q_upper))

    # ---------------------------------------------------------
    # Combined optimization vector
    #
    # z = [m1, m2, ..., log(q1), log(q2), ...]
    # ---------------------------------------------------------
    z0 = np.concatenate([
        m0,
        log_q0,
    ])

    lo_combined = np.concatenate([
        lo_scaled,
        log_q_lo,
    ])

    hi_combined = np.concatenate([
        hi_scaled,
        log_q_hi,
    ])

    n_theta = len(theta0)

    def nll_scaled(z):

        # Thermal parameters
        m = z[:n_theta]
        theta = m * theta0

        # Process-noise variances
        log_q = z[n_theta:]
        q = np.exp(log_q)
        Q_candidate = np.diag(q)

        return neg_log_likelihood(
            theta,
            df_train,
            C,
            Q_candidate,
            R,
        )

    initial_nll = nll_scaled(z0)

    # ---------------------------------------------------------
    # Find theta and Q that minimize the likelihood cost
    # ---------------------------------------------------------
    result = minimize_nll(
        nll_scaled,
        z0,
        lo_combined,
        hi_combined,
    )

    # ---------------------------------------------------------
    # Convert optimized parameters back to physical units
    # ---------------------------------------------------------
    theta_hat = result.x[:n_theta] * theta0

    log_q_hat = result.x[n_theta:]
    q_hat = np.exp(log_q_hat)
    Q_hat = np.diag(q_hat)

    final_nll = nll_scaled(result.x)

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------
    print("Optimization success:", result.success)
    print("Optimizer message:", result.message)

    print("\nInitial parameters:", theta0)
    print("Estimated parameters:", theta_hat)
    print("Parameter bounds (lo, hi):", (lo, hi))

    print("\nInitial Q:")
    print(Q0)

    print("Estimated Q:")
    print(Q_hat)

    print("\nInitial NLL:", initial_nll)
    print("Final NLL:", final_nll)
    print("NLL reduction:", initial_nll - final_nll)

    # ---------------------------------------------------------
    # Check thermal parameter bounds
    # ---------------------------------------------------------
    tol = 1e-3

    at_lo = np.abs(theta_hat - lo) / np.abs(lo) < tol
    at_hi = np.abs(theta_hat - hi) / np.abs(hi) < tol

    if at_lo.any() or at_hi.any():
        stuck = [
            name
            for name, flag in zip(
                PARAMETER_NAMES,
                at_lo | at_hi
            )
            if flag
        ]

        print(
            f"WARNING: parameters pinned at bounds: "
            f"{stuck} -- widen and re-run"
        )

    # ---------------------------------------------------------
    # Check Q bounds
    # ---------------------------------------------------------
    q_at_lo = q_hat <= q_lower * (1.0 + tol)
    q_at_hi = q_hat >= q_upper * (1.0 - tol)

    if q_at_lo.any() or q_at_hi.any():
        stuck_q = [
            f"q{i + 1}"
            for i, flag in enumerate(q_at_lo | q_at_hi)
            if flag
        ]

        print(
            f"WARNING: Q parameters pinned at bounds: "
            f"{stuck_q} -- inspect bounds"
        )

    return theta_hat, Q_hat, result