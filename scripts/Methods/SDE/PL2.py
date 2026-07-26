import numpy as np
from scipy.optimize import minimize
from SDE_config import THETA0, LOWER_BOUND, UPPER_BOUND, Q, R, C,PARAMETER_NAMES
from parameter_estimation import neg_log_likelihood

def run_pl2(name_a, name_b, df_train, theta_hat, nll_ref, n_points=15, inner_maxiter=120):
    i, j = PARAMETER_NAMES.index(name_a), PARAMETER_NAMES.index(name_b)

    if len(theta_hat) > 2:
        # 3+ params: fixing 2 leaves free params to re-optimize
        gi, gj, Z = profile_likelihood_2d(
            df_train, theta_hat, nll_ref, i, j,
            n_points=n_points, inner_maxiter=inner_maxiter)
    else:
        # exactly 2 params: no free params, evaluate the surface directly
        gi, gj, Z = likelihood_surface_2d(
            df_train, theta_hat, nll_ref, i, j)

    plot_pl2(gi, gj, Z, name_a, name_b, theta_hat[i], theta_hat[j])
    return name_a, name_b, gi, gj, Z


#EQ 22 in Paper:_ PL2(θi, θj) = min over all other params  g(θ; θi fixed, θj fixed)
def profile_likelihood_2d(df, theta_hat, nll_ref, i, j,
                          n_points=15, inner_maxiter=150):
    """
    PL2: fix parameters i and j on a 2D grid, re-optimize the rest at each
    grid point. Returns (grid_i, grid_j, Z) where Z[a,b] = nll - nll_ref.
    """
    theta_hat = np.asarray(theta_hat, float)
    lo = np.asarray(LOWER_BOUND, float)
    hi = np.asarray(UPPER_BOUND, float)
    n = len(theta_hat)

    # grids for the two held parameters (±50% around optimum, adjust as needed)
    gi = np.linspace(max(lo[i], 0.5*theta_hat[i]), min(hi[i], 1.5*theta_hat[i]), n_points)
    gj = np.linspace(max(lo[j], 0.5*theta_hat[j]), min(hi[j], 1.5*theta_hat[j]), n_points)

    free = [k for k in range(n) if k not in (i, j)]
    scale = theta_hat[free]
    bounds = list(zip(lo[free]/scale, hi[free]/scale))

    Z = np.full((n_points, n_points), np.nan)

    for a, vi in enumerate(gi):
        m0 = np.ones(len(free))               # warm-start per row
        for b, vj in enumerate(gj):
            def obj(m_free):
                theta = theta_hat.copy()
                theta[free] = m_free * scale
                theta[i] = vi
                theta[j] = vj
                val = neg_log_likelihood(theta, df, C, Q, R)
                return val if np.isfinite(val) else 1e10
            res = minimize(obj, m0, method="Powell",
                           bounds=bounds, options={"maxfev": inner_maxiter})
            Z[a, b] = res.fun - nll_ref
            m0 = res.x                          # warm-start next cell
        print(f"PL2 {PARAMETER_NAMES[i]} vs {PARAMETER_NAMES[j]}: row {a+1}/{n_points}")

    return gi, gj, Z


import matplotlib.pyplot as plt
from scipy.stats import chi2

def plot_pl2(gi, gj, Z, name_i, name_j, theta_hat_i, theta_hat_j):
    levels = [chi2.ppf(p, 2) for p in (0.90, 0.95, 0.99)]  # 4.61, 5.99, 9.21

    fig, ax = plt.subplots(figsize=(6, 5))
    # cap Z so the color scale focuses on the low-NLL (interesting) region
    Zc = np.minimum(Z, 20)
    im = ax.contourf(gj, gi, Zc, levels=30, cmap="turbo")
    cs = ax.contour(gj, gi, Z, levels=levels, colors="k", linewidths=1)
    ax.clabel(cs, fmt={levels[0]:"90%", levels[1]:"95%", levels[2]:"99%"})
    ax.plot(theta_hat_j, theta_hat_i, "w*", markersize=2, label="optimum")

    ax.set_xlabel(name_j)
    ax.set_ylabel(name_i)
    ax.set_title(f"PL2: {name_i} vs {name_j}")
    fig.colorbar(im, label=r"$g(\theta)-g(\hat\theta)$")
    ax.legend()
    plt.tight_layout()
    plt.show()

def likelihood_surface_2d(df, theta_hat, nll_ref, i, j, n_points=25):
    lo, hi = np.asarray(LOWER_BOUND, float), np.asarray(UPPER_BOUND, float)
    gi = np.linspace(0.7*theta_hat[i], 1.3*theta_hat[i], n_points)
    gj = np.linspace(0.7*theta_hat[j], 1.3*theta_hat[j], n_points)
    Z = np.full((n_points, n_points), np.nan)
    for a, vi in enumerate(gi):
        for b, vj in enumerate(gj):
            theta = theta_hat.copy()
            theta[i] = vi; theta[j] = vj
            val = neg_log_likelihood(theta, df, C, Q, R)
            Z[a, b] = val - nll_ref
    return gi, gj, Z