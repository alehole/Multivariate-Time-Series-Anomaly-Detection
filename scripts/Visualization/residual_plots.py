from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import Tensor

def _plot_residual_series(
    time_values,
    residual: np.ndarray,
    target_cols: list[str],
    axes=None,
    anomaly_masks: list[np.ndarray] | None = None,
    label: str = "Residual",
    anomaly_threshold: float | list[float] | np.ndarray = 3.0,
    show_threshold: bool = True,
    show_legend: bool = False,
    plot_targets: list[str] | None = None,
):
    if plot_targets is None:
        plot_targets = target_cols

    plot_indices = [
        target_cols.index(col)
        for col in plot_targets
    ]

    created_fig = False

    if axes is None:
        fig, axes = plt.subplots(
            len(plot_targets),
            1,
            figsize=(14, 3 * len(plot_targets)),
            sharex=True,
        )

        if len(plot_targets) == 1:
            axes = [axes]

        created_fig = True

    thresholds = np.atleast_1d(anomaly_threshold)

    for ax_idx, target_idx in enumerate(plot_indices):
        col = target_cols[target_idx]
        ax = axes[ax_idx]

        y = residual[:, target_idx]

        threshold_i = (
            thresholds[0]
            if len(thresholds) == 1
            else thresholds[target_idx]
        )

        ax.plot(time_values, y, label=label)
        ax.axhline(0.0, linestyle="--", color="black")
        ax.set_ylabel("Residual ΔT (°C)")

        if show_threshold:
            ax.set_title(
                f"Prediction residual – {col} | "
                f"threshold = ±{threshold_i:.1f} °C"
            )
            ax.axhline(threshold_i, linestyle=":", color="red")
            ax.axhline(-threshold_i, linestyle=":", color="red")
        else:
            ax.set_title(f"Prediction residual – {col}")

        ax.grid(True)

        if anomaly_masks is not None:
            mask = anomaly_masks[target_idx]

            ax.scatter(
                np.asarray(time_values)[mask],
                y[mask],
                color="red",
                s=10,
                label="Anomaly" if show_legend else None,
            )

        if show_legend:
            ax.legend()

    axes[-1].set_xlabel("Time")

    if created_fig:
        plt.tight_layout()
        plt.show()

def plot_residuals_inference(
    data: pd.DataFrame,
    residual: np.ndarray,
    target_cols: list[str],
    ts_col: str = "Created",
    anomaly_threshold: float | list[float] | np.ndarray = 3.0,
    show_threshold: bool = True,
    plot_targets: list[str] | None = None,
):
    t = data[ts_col].to_numpy()

    anomaly_masks = [
        data[f"{col}_anomaly"].to_numpy(dtype=bool)
        for col in target_cols
    ]

    _plot_residual_series(
        time_values=t,
        residual=residual,
        target_cols=target_cols,
        anomaly_masks=anomaly_masks,
        label="Residual",
        anomaly_threshold=anomaly_threshold,
        show_threshold=show_threshold,
        show_legend=False,
        plot_targets=plot_targets,
    )

def plot_residuals_profiles(
    data: pd.DataFrame,
    pred_c: np.ndarray,
    test_mask: Tensor,
    test_profiles: list[int],
    target_cols: list[str],
    y_scaler,
    ts_col: str = "Created",
    anomaly_threshold: float = 3.0,
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

        residual = y_true - y_pred

        thresholds = np.atleast_1d(anomaly_threshold)

        anomaly_masks = [
            np.abs(residual[:, j]) >
            (thresholds[0] if len(thresholds) == 1 else thresholds[j])
            for j in range(len(target_cols))
        ]

        _plot_residual_series(
            time_values=t,
            residual=residual,
            target_cols=target_cols,
            axes=axes,
            anomaly_masks=anomaly_masks,
            label=f"Profile {pid}" if len(test_profiles) > 1 else "Residual",
            anomaly_threshold=anomaly_threshold,
            show_legend=(len(test_profiles) > 1 and i == 0),
        )

    plt.tight_layout()
    plt.show()