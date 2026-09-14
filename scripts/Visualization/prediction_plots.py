from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import Tensor
from Visualization.metrics  import compute_metrics, add_metrics_box


def _plot_actual_vs_predicted_scatter(
    actual: np.ndarray,
    predicted: np.ndarray,
    target_cols: list[str],
    title: str | None = None,
):
    fig, axes = plt.subplots(1, len(target_cols), figsize=(5 * len(target_cols), 5))

    if len(target_cols) == 1:
        axes = [axes]

    for j, col in enumerate(target_cols):
        ax = axes[j]

        y_true = actual[:, j]
        y_pred = predicted[:, j]

        ax.scatter(y_true, y_pred, s=8, alpha=0.5)

        mn = min(np.nanmin(y_true), np.nanmin(y_pred))
        mx = max(np.nanmax(y_true), np.nanmax(y_pred))
        ax.plot([mn, mx], [mn, mx], color="black", linestyle="--")

        metrics = compute_metrics(y_true, y_pred)
        add_metrics_box(ax, metrics)

        ax.set_title(col)
        ax.set_xlabel("Actual Temperature (°C)")
        ax.set_ylabel("Predicted Temperature (°C)")
        ax.grid(True)
        ax.set_aspect("equal", adjustable="box")

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()

def plot_predicted_vs_actual_profiles(
    data: pd.DataFrame,
    pred_c: np.ndarray,
    test_mask: Tensor,
    test_profiles: list[int],
    target_cols: list[str],
    y_scaler,
    ts_col: str = "Created",
):
    plot_df = data.loc[:, [ts_col, "profile_id"] + target_cols].copy()
    test_df = plot_df.loc[
        plot_df["profile_id"].isin(test_profiles),
        [ts_col, "profile_id"] + target_cols
    ].copy()
    test_df = test_df.sort_values(["profile_id", ts_col])

    y_true_all = []
    y_pred_all = []

    for i, (pid, y_df) in enumerate(test_df.groupby("profile_id", sort=False)):
        y_true = y_scaler.inverse_transform(
            y_df[target_cols].reset_index(drop=True).to_numpy()
        )

        y_pred = pred_c[i, :len(y_true), :]
        mask_i = test_mask[i, :len(y_true)].cpu().numpy().astype(bool)

        y_true_all.append(y_true[mask_i])
        y_pred_all.append(y_pred[mask_i])

    y_true_all = np.vstack(y_true_all)
    y_pred_all = np.vstack(y_pred_all)

    _plot_actual_vs_predicted_scatter(
        actual=y_true_all,
        predicted=y_pred_all,
        target_cols=target_cols,
        title="Predicted vs Actual Temperatures",
    )

def plot_predicted_vs_actual_inference(
    actual_df: pd.DataFrame,
    predicted: np.ndarray,
    target_cols: list[str],
):
    actual = actual_df[target_cols].to_numpy()

    _plot_actual_vs_predicted_scatter(
        actual=actual,
        predicted=predicted,
        target_cols=target_cols,
        title="Predicted vs Actual Temperatures",
    )



def plot_actual_vs_predicted(
    actual_df: pd.DataFrame,
    predicted: np.ndarray,
    target_cols: list[str],
    ts_col: str = "Created",
    anomaly_threshold: float = 3.0,
):

    t = actual_df[ts_col]
    fig, axes = plt.subplots(len(target_cols), 1, figsize=(14, 8), sharex=True)

    if len(target_cols) == 1:
        axes = [axes]

    for i, col in enumerate(target_cols):
        y_true = actual_df[col].to_numpy()
        y_pred = predicted[:, i]

        ax = axes[i]

        ax.plot(t, y_true, label="Actual", linewidth=2)
        ax.plot(t, y_pred, label="Predicted", linestyle="--")

        metrics = compute_metrics(y_true, y_pred)
        add_metrics_box(ax, metrics)

        ax.set_ylabel("Temp (°C)")
        ax.set_title(col)
        ax.grid(True)
        anomaly_mask = np.abs(y_true - y_pred) > anomaly_threshold

        ax.scatter(
            t[anomaly_mask],
            y_true[anomaly_mask],
            color="red",
            s=15,
            label="Anomaly" if i == 0 else None,
        )

        if i == 0:
            ax.legend(loc="lower right")

    axes[-1].set_xlabel("Time")

    plt.tight_layout()
    plt.show()


def plot_profile_predictions(
    data: pd.DataFrame,
    pred_c: np.ndarray,
    test_mask: Tensor,
    test_profiles: list[int],
    target_cols: list[str],
    y_scaler,
):
    """
    Plot predicted vs actual temperatures for each test profile.

    Each row corresponds to a profile (time sequence) and each column
    corresponds to a target temperature variable.

    Parameters
    ----------
    data : pd.DataFrame
        Original dataset containing true target values.

    pred_c : np.ndarray
        Model predictions in original temperature units (°C),
        shape (B, T, S).

    test_mask : Tensor
        Boolean mask indicating valid timesteps (used to ignore padding).

    test_profiles : list[int]
        Profile IDs used as test sequences.

    target_cols : list[str]
        Names of predicted temperature targets.

    y_scaler :
        Scaler used during training for target normalization.
        Used here to convert true values back to °C.
    """
    grouped = data.loc[
        data["profile_id"].isin(test_profiles),
        target_cols + ["profile_id"]
    ].groupby("profile_id", sort=False)

    fig, axes = plt.subplots(
        len(test_profiles),
        len(target_cols),
        figsize=(20, 10),
        squeeze=False
    )

    for i, (pid, y_df) in enumerate(grouped):
        y_true = y_scaler.inverse_transform(
            y_df[target_cols].reset_index(drop=True).to_numpy()
        )

        y_pred = pred_c[i, :len(y_true), :]

        mask_i = test_mask[i, :len(y_true)].cpu().numpy().astype(bool)
        y_true = y_true[mask_i]
        y_pred = y_pred[mask_i]

        for j, col in enumerate(target_cols):
            ax = axes[i, j]
            ax.plot(y_true[:, j], label="Actual")
            ax.plot(y_pred[:, j], label="Prediction")

            metrics = compute_metrics(y_true[:, j], y_pred[:, j])

            add_metrics_box(ax, metrics)

            if j == 0:
                ax.set_ylabel(f"Profile {pid}\nTemp (°C)")
            if i == 0:
                ax.set_title(col)
            if i == len(test_profiles) - 1:
                ax.set_xlabel("Steps")

    axes[0, 0].legend()
    plt.tight_layout()
    plt.show()


def plot_time_predictions(
    data: pd.DataFrame,
    pred_c: np.ndarray,
    test_mask: Tensor,
    test_profiles: list[int],
    target_cols: list[str],
    y_scaler,
    ts_col: str = "Created",
):
    plot_df = data.loc[:, [ts_col, "profile_id"] + target_cols].copy()
    test_df = plot_df.loc[
        plot_df["profile_id"].isin(test_profiles),
        [ts_col, "profile_id"] + target_cols
    ].copy()
    test_df = test_df.sort_values(["profile_id", ts_col])

    fig, axes = plt.subplots(len(target_cols), 1, figsize=(16, 9), sharex=True)
    if len(target_cols) == 1:
        axes = [axes]

    for i, (pid, y_df) in enumerate(test_df.groupby("profile_id", sort=False)):
        y_true = y_scaler.inverse_transform(
            y_df[target_cols].reset_index(drop=True).to_numpy()
        )

        y_pred = pred_c[i, :len(y_true), :]
        mask_i = test_mask[i, :len(y_true)].cpu().numpy().astype(bool)

        y_true = y_true[mask_i]
        y_pred = y_pred[mask_i]
        t = y_df[ts_col].to_numpy()[mask_i]

        for j, col in enumerate(target_cols):
            ax = axes[j]
            ax.plot(t, y_true[:, j], label="Actual" if i == 0 else None)
            ax.plot(t, y_pred[:, j], label="Prediction" if i == 0 else None)
            ax.set_title(col)
            ax.set_ylabel("Temp (°C)")

    axes[-1].set_xlabel("Time")
    axes[0].legend()
    plt.tight_layout()
    plt.show()
