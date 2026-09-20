import numpy as np
from scipy.optimize import minimize
from SDE_config import THETA0, LOWER_BOUND, UPPER_BOUND, Q, R, C,PARAMETER_NAMES
from parameter_estimation import neg_log_likelihood, minimize_nll

def run_pl1(df_train, theta_hat, nll_ref):
    ## PL1
    profiles = profile_likelihood_1d(
        df_train, theta_hat, nll_ref,
        n_points=21,  # odd number so the optimum sits on a grid point
        inner_maxiter=150,  # keep inner solves short
    )
    import numpy as np
    for name, (grid, rel) in profiles.items():
        finite = np.isfinite(rel)

        print(f"\n{name}")
        print("Minimum relative likelihood:", np.min(rel[finite]))
        print("Maximum relative likelihood:", np.max(rel[finite]))
        print(
            "Grid value at minimum:",
            grid[finite][np.argmin(rel[finite])]
        )

    plot_profile_likelihood(profiles, ndf=1, y_limit=7.5)
    return profiles

#EQ 19 in Paper:
def profile_likelihood_1d(df, theta_hat, nll_ref,
                          n_points=25, inner_maxiter=300):
    """
    PL1: for each parameter, sweep it across [LOWER, UPPER], re-optimizing
    the remaining parameters at each step. Returns {name: (grid, rel_nll)}
    where rel_nll = nll(theta_i fixed) - nll_ref, directly comparable to
    chi^2(alpha, ndf=1) thresholds.
    """
    theta_hat = np.asarray(theta_hat, float)
    lo = np.asarray(LOWER_BOUND, float)
    hi = np.asarray(UPPER_BOUND, float)
    n = len(theta_hat)
    profiles = {}

    for i in range(n):
        center = theta_hat[i]
        span = 0.03 * center  # ±2% around the optimum
        #span = 0.5 * center  # ±2% around the optimum
        grid = np.linspace(max(lo[i], center - span),
                           min(hi[i], center + span), n_points)
        rel = np.full(n_points, np.nan)

        free = [j for j in range(n) if j != i]
        scale = theta_hat[free]                 # scale free params by the optimum
        bounds = list(zip(lo[free] / scale, hi[free] / scale))

        def solve(val, m0):
            def obj(m_free):
                theta = theta_hat.copy()
                theta[free] = m_free * scale
                theta[i] = val
                return neg_log_likelihood(theta, df, C, Q, R)
            res = minimize(obj, m0, method="Powell",
                           bounds=bounds, options={"maxiter": inner_maxiter})
            return res.fun - nll_ref, res.x

        # two-sided sweep outward from the free optimum, warm-starting each step
        i0 = int(np.argmin(np.abs(grid - theta_hat[i])))
        m0 = np.ones(len(free))
        for g in range(i0, n_points):           # right
            rel[g], m0 = solve(grid[g], m0)
        m0 = np.ones(len(free))
        for g in range(i0 - 1, -1, -1):         # left
            rel[g], m0 = solve(grid[g], m0)

        profiles[PARAMETER_NAMES[i]] = (grid, rel)
        print(f"profiled {PARAMETER_NAMES[i]}")

    return profiles

import matplotlib.pyplot as plt
from scipy.stats import chi2

def plot_profile_likelihood(profiles, ndf=1, y_limit=12.0):
    levels = {
        "90%": chi2.ppf(0.90, ndf),
        "95%": chi2.ppf(0.95, ndf),
        "99%": chi2.ppf(0.99, ndf),
    }

    for name, (grid, rel) in profiles.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(grid, rel, "b-", linewidth=1.5, label="Profile likelihood")

        for label, t in levels.items():
            if t <= y_limit:
                ax.axhline(t, color="k", lw=0.7)
                ax.text(grid[0], t, f" {label}", fontsize=8, va="bottom")

        ax.set_xlabel(name)
        ax.set_ylabel(r"$g(\theta_i)-g(\hat{\theta})$")
        ax.set_title(f"Profile likelihood for {name}")

        ax.set_ylim(0, y_limit)
        ax.set_xlim(grid.min(), grid.max())   # <-- per-parameter, from its own grid
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()