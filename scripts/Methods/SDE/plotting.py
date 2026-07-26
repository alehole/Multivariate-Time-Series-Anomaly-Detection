import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
from SDE_config import STATE_COLS, MEAS_COLS, NIS_THRESHOLD_PERCENTILE
import pandas as pd

def get_plot_cols():
    plot_cols = MEAS_COLS
    plot_indices = [STATE_COLS.index(col) for col in plot_cols]
    return plot_cols, plot_indices


def print_metrics(df, x_hat, title=""):
    print(f"\n{title} metrics")

    plot_cols, plot_indices = get_plot_cols()

    for name, idx in zip(plot_cols, plot_indices):
        y_true = df[name].values
        y_pred = x_hat[:, idx]

        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"{name}: MAE={mae:.3f} °C, R²={r2:.4f}")


def plot_results(df, x_hat, P_cov, title=""):
    plot_cols, plot_indices = get_plot_cols()
    std = np.sqrt(np.diagonal(P_cov, axis1=1, axis2=2))

    fig, axes = plt.subplots(
        len(plot_cols),
        1,
        figsize=(14, 3 * len(plot_cols)),
        sharex=True
    )

    if len(plot_cols) == 1:
        axes = [axes]

    for ax, name, idx in zip(axes, plot_cols, plot_indices):
        ax.plot(df["Created"], df[name], label="Measurement")
        ax.plot(df["Created"], x_hat[:, idx], label="EKF posterior estimate")

        ax.fill_between(
            df["Created"],
            x_hat[:, idx] - 2 * std[:, idx],
            x_hat[:, idx] + 2 * std[:, idx],
            alpha=0.35,
            label="95% uncertainty interval",
        )

        ax.set_ylabel("Temperature [°C]")
        ax.set_title(f"{name} State Estimation")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_innovations(df, innovations, title=""):
    plot_cols, plot_indices = get_plot_cols()

    fig, axes = plt.subplots(
        len(plot_cols),
        1,
        figsize=(14, 3 * len(plot_cols)),
        sharex=True
    )

    if len(plot_cols) == 1:
        axes = [axes]

    for ax, name, idx in zip(axes, plot_cols, range(len(plot_cols))):
        ax.plot(df["Created"], innovations[:, idx], label="Innovation")
        ax.axhline(0, linestyle="--", label="Zero reference")

        ax.set_ylabel("Residual [°C]")
        ax.set_title(f"{name} Innovation Residual")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_NIS(df, NIS, title=""):
    from scipy.stats import chi2
    threshold = chi2.ppf(0.995, df=len(MEAS_COLS))
    anomaly = NIS > threshold
    plt.figure(figsize=(14, 4))
    plt.plot(df["Created"], NIS, label="Normalized Innovation Squared (NIS)")

    plt.scatter(
        df["Created"][anomaly],
        NIS[anomaly],
        color="red",
        s=20,
        label="Potential anomaly",
    )

    plt.axhline(
        threshold,
        linestyle="--",
        label=f"99.5% threshold = {threshold:.2f}",
    )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Anomaly score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_full_ekf_comparison(df, x_sim, x_pred_hist, x_hat, title=""):
    plot_cols, plot_indices = get_plot_cols()

    fig, axes = plt.subplots(
        len(plot_cols),
        1,
        figsize=(14, 3 * len(plot_cols)),
        sharex=True
    )

    if len(plot_cols) == 1:
        axes = [axes]

    for ax, name, idx in zip(axes, plot_cols, plot_indices):
        ax.plot(
            df["Created"],
            df[name],
            color="black",
            linewidth=1.2,
            label="Measurement",
        )

        ax.plot(
            df["Created"],
            x_sim[:, idx],
            color="blue",
            linewidth=1.2,
            label="Open-loop simulation",
        )

        ax.plot(
            df["Created"],
            x_pred_hist[:, idx],
            color="orange",
            linestyle="--",
            linewidth=1.2,
            label="EKF prior estimate",
        )

        ax.plot(
            df["Created"],
            x_hat[:, idx],
            color="green",
            linewidth=2,
            alpha=0.6,
            label="EKF posterior estimate",
        )

        ax.set_ylabel("Temperature [°C]")
        ax.set_title(f"{name} Comparison")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_simulated_vs_actual(df, x_sim, title="Open-loop simulation vs measurement"):
    plot_cols, plot_indices = get_plot_cols()

    fig, axes = plt.subplots(
        len(plot_cols),
        1,
        figsize=(14, 3 * len(plot_cols)),
        sharex=True
    )

    if len(plot_cols) == 1:
        axes = [axes]

    for ax, name, idx in zip(axes, plot_cols, plot_indices):
        ax.plot(
            df["Created"],
            df[name],
            color="black",
            linewidth=1.4,
            label="Measurement",
        )

        ax.plot(
            df["Created"],
            x_sim[:, idx],
            color="blue",
            linewidth=1.4,
            linestyle="--",
            label="Open-loop simulation",
        )

        ax.set_ylabel("Temperature [°C]")
        ax.set_title(f"{name}: Simulated vs Actual")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

    from scipy.stats import chi2  # move to top of file

def plot_residuals_vs_time(df_train, x_sim_train, df_test, x_sim_test,
                           window="6h", title="Open-loop residual"):
    """Residual (measured - simulated) over the full record, train and test
    on a single time axis, with a rolling mean to expose slow drift."""
    plot_cols, plot_indices = get_plot_cols()

    fig, axes = plt.subplots(len(plot_cols), 1, figsize=(14, 3.2 * len(plot_cols)),
                             sharex=True)
    if len(plot_cols) == 1:
        axes = [axes]

    for ax, name, idx in zip(axes, plot_cols, plot_indices):
        rows = []
        for df, x_sim, split in ((df_train, x_sim_train, "train"),
                                 (df_test, x_sim_test, "test")):
            rows.append(pd.DataFrame({
                "t": df["Created"].values,
                "r": df[name].values - x_sim[:, idx],
                "split": split,
            }))
        res = pd.concat(rows).sort_values("t").set_index("t")

        for split, colour in (("train", "tab:blue"), ("test", "tab:red")):
            sub = res[res["split"] == split]
            ax.plot(sub.index, sub["r"], colour, lw=0.5, alpha=0.30)
            ax.plot(sub.index, sub["r"].rolling(window).mean(),
                    colour, lw=2.0, label=f"{split} ({window} mean)")

        mu_tr = res.loc[res["split"] == "train", "r"].mean()
        ax.axhline(0.0, color="k", lw=1.0)
        ax.axhline(mu_tr, color="k", lw=0.8, ls=":",
                   label=f"train mean = {mu_tr:+.2f} °C")

        # boundary between the two periods
        ax.axvline(df_test["Created"].iloc[0], color="k", lw=1.0, ls="--", alpha=0.6)

        ax.set_ylabel(f"{name} residual [°C]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Time")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()