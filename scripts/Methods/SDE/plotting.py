import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from SDE_config import STATE_COLS, MEAS_COLS, NIS_THRESHOLD_PERCENTILE
import pandas as pd

def get_plot_cols():
    plot_cols = MEAS_COLS
    plot_indices = [STATE_COLS.index(col) for col in plot_cols]
    return plot_cols, plot_indices


def print_metrics(
    df,
    estimates,
    title: str = "",
    skip_initial: int = 0,
) -> None:
    """
    Calculate prediction metrics for all measured model states.
    """

    estimates = np.asarray(estimates, dtype=float)

    if len(df) != len(estimates):
        raise ValueError(
            "Measurement and estimate lengths do not match: "
            f"measurements={len(df)}, estimates={len(estimates)}."
        )

    if skip_initial < 0 or skip_initial >= len(df):
        raise ValueError(
            f"skip_initial must be between 0 and {len(df) - 1}."
        )

    plot_cols, state_indices = get_plot_cols()

    print(f"\n{title} metrics")

    for column, state_idx in zip(
        plot_cols,
        state_indices,
    ):
        if column not in df.columns:
            raise KeyError(
                f"Measurement column '{column}' was not found."
            )

        if state_idx >= estimates.shape[1]:
            raise IndexError(
                f"State index {state_idx} is outside estimate "
                f"shape {estimates.shape}."
            )

        y_true = (
            df[column]
            .to_numpy(dtype=float)[skip_initial:]
        )

        y_pred = estimates[
            skip_initial:,
            state_idx,
        ]

        # Ignore rows containing missing or non-finite values.
        valid = (
            np.isfinite(y_true)
            & np.isfinite(y_pred)
        )

        y_true_valid = y_true[valid]
        y_pred_valid = y_pred[valid]

        if len(y_true_valid) == 0:
            print(f"{column}: no valid observations")
            continue

        residual = y_true_valid - y_pred_valid

        mae = mean_absolute_error(
            y_true_valid,
            y_pred_valid,
        )

        mse = mean_squared_error(
            y_true_valid,
            y_pred_valid,
        )

        rmse = np.sqrt(mse)
        max_abs = np.max(np.abs(residual))

        if len(y_true_valid) >= 2:
            r2 = r2_score(
                y_true_valid,
                y_pred_valid,
            )
        else:
            r2 = np.nan

        print(
            f"{column}: "
            f"MAE={mae:.3f} °C, "
            f"RMSE={rmse:.3f} °C, "
            f"MaxAbs={max_abs:.3f} °C, "
            f"R²={r2:.4f}, "
            f"N={len(y_true_valid):,}"
        )


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


def plot_full_ekf_comparison(
    df,
    x_sim,
    x_pred_hist,
    x_hat,
    title="",
    target_cols=None,
):
    plot_cols, plot_indices = get_plot_cols()

    # Only keep requested states
    if target_cols is not None:
        selected = [
            (name, idx)
            for name, idx in zip(plot_cols, plot_indices)
            if name in target_cols
        ]

        plot_cols = [name for name, _ in selected]
        plot_indices = [idx for _, idx in selected]

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