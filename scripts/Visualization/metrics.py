from __future__ import annotations
import numpy as np

def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
) -> dict[str, float]:

    residual = y_true - y_pred
    mse = np.nanmean(residual ** 2)
    rmse = np.sqrt(mse)
    mae = np.nanmean(np.abs(residual))
    mx = np.nanmax(np.abs(residual))

    # R2
    ss_res = np.nansum(residual ** 2)
    ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - ss_res / ss_tot

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "max_abs": float(mx),
        "r2": float(r2)
    }

def add_metrics_box(
    ax,
    metrics: dict[str, float],
    x: float = 0.02,
    y: float = 0.95,
) -> None:
    ax.text(
        x,
        y,
        f"MSE: {metrics['mse']:.2f} K²\n"
        f"MAE: {metrics['mae']:.2f} °C\n"
        f"RMSE: {metrics['rmse']:.2f} °C\n"
        f"max |ΔT|: {metrics['max_abs']:.1f} °C\n"
        f"R²: {metrics['r2']:.3f}",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )