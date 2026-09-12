from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config as cfg
from Methods.Blackbox.profile_dataset import (
    create_profiles,
    tensorize_profiles,
)
from Methods.TNN.tnn_model import build_model
from Visualization.prediction_plots import (
    plot_actual_vs_predicted,
    plot_predicted_vs_actual_inference,
)
from Visualization.residual_plots import plot_residuals_inference
from scripts.misc.feature_engineering import ts_cols


ANOMALY_THRESHOLD = 5.0
USE_ANOMALY_FILE = False


def load_tnn(
    path: str | Path,
    device: torch.device,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    required_keys = {
        "model_state_dict",
        "input_cols",
        "target_cols",
        "temperature_cols",
        "cooling_columns",
        "dt_s",
        "window_steps",
        "n_neurons",
        "x_scaler",
        "y_scaler",
    }

    missing = required_keys.difference(checkpoint)
    if missing:
        raise KeyError(
            "TNN checkpoint is missing keys: "
            + ", ".join(sorted(missing))
        )

    model = build_model(
        dt_s=checkpoint["dt_s"],
        input_cols=checkpoint["input_cols"],
        target_cols=checkpoint["target_cols"],
        temperature_cols=checkpoint["temperature_cols"],
        cooling_columns=checkpoint["cooling_columns"],
        n_neurons=checkpoint["n_neurons"],
        device=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    return model, checkpoint


def load_and_prepare_test_data(
    csv_path: str | Path,
    input_cols: list[str],
    target_cols: list[str],
    x_scaler,
    y_scaler,
    checkpoint_dt_s: float,
    window_steps: int,
    device: torch.device,
):
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {csv_path}"
        )

    data = pd.read_csv(csv_path)
    data, data_dt_s = ts_cols(data, cfg.TS_COL)

    if not np.isclose(
        data_dt_s,
        checkpoint_dt_s,
        rtol=0.05,
    ):
        print(
            "Warning: sampling interval differs from training: "
            f"test={data_dt_s:.2f} s, "
            f"training={checkpoint_dt_s:.2f} s"
        )

    label_cols = [
        col
        for col in ["fault_label", "fault_type", "fault_id"]
        if col in data.columns
    ]

    required_generic = set(
        [*input_cols, *target_cols]
    )

    raw_model_cols = [
        raw_col
        for raw_col, generic_col in cfg.RENAME_MAP.items()
        if generic_col in required_generic
    ]

    required_raw_cols = [cfg.TS_COL, *raw_model_cols]

    missing_raw_cols = [
        col
        for col in required_raw_cols
        if col not in data.columns
    ]

    if missing_raw_cols:
        raise KeyError(
            "The test dataset is missing required columns: "
            f"{missing_raw_cols}"
        )

    keep_cols = list(
        dict.fromkeys(
            [*required_raw_cols, *label_cols]
        )
    )

    data = data[keep_cols].copy()
    data = data.rename(columns=cfg.RENAME_MAP)

    data[input_cols] = data[input_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )
    data[target_cols] = data[target_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    data, test_profiles = create_profiles(
        data,
        ts_col=cfg.TS_COL,
        window_steps=window_steps,
        dt_s=data_dt_s,
    )

    # Preserve the true measurements, including target dropouts, before
    # temporary inference preprocessing.
    actual_cols = [
        cfg.TS_COL,
        "profile_id",
        *target_cols,
        *label_cols,
    ]
    actual_df = data[actual_cols].copy()

    # Match the training preprocessing for model inputs.
    data[input_cols] = (
        data.groupby("profile_id")[input_cols]
        .transform(lambda g: g.ffill().bfill())
    )

    # The TNN uses the first target value of every profile as its initial
    # thermal state. Fill only the temporary model copy; actual_df above
    # remains unchanged so target dropout is still detected directly.
    data[target_cols] = (
        data.groupby("profile_id")[target_cols]
        .transform(lambda g: g.ffill().bfill())
    )

    data[input_cols] = x_scaler.transform(
        data[input_cols]
    )
    data[target_cols] = y_scaler.transform(
        data[target_cols]
    )

    x_test, y_test, mask_test = tensorize_profiles(
        data,
        test_profiles,
        input_cols,
        target_cols,
        device=device,
    )

    test_tensor = torch.cat(
        [x_test, y_test],
        dim=2,
    )

    return (
        data,
        actual_df,
        test_profiles,
        test_tensor,
        mask_test,
    )


def predict_tnn(
    model,
    test_tensor: torch.Tensor,
    test_mask: torch.Tensor,
    input_cols: list[str],
    target_cols: list[str],
    y_scaler,
):
    n_in = len(input_cols)
    n_out = len(target_cols)

    x = test_tensor[:, :-1, :n_in]
    state0 = test_tensor[:, 0, -n_out:]

    pair_mask = (
        test_mask[:, :-1]
        & test_mask[:, 1:]
    )

    model.eval()
    with torch.no_grad():
        pred_scaled, _ = model(x, state0)

    pred_scaled = pred_scaled.detach().cpu().numpy()
    batch_size, seq_len, n_features = pred_scaled.shape

    predicted_c = y_scaler.inverse_transform(
        pred_scaled.reshape(-1, n_features)
    ).reshape(batch_size, seq_len, n_features)

    return predicted_c, pair_mask


def flatten_aligned_results(
    actual_df: pd.DataFrame,
    test_profiles: list[int],
    predicted_c: np.ndarray,
    pair_mask: torch.Tensor,
    target_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Align prediction T(k+1) with the measured row at k+1."""

    mask_np = pair_mask.detach().cpu().numpy().astype(bool)

    blocks = []
    pred_blocks = []

    grouped = actual_df.groupby(
        "profile_id",
        sort=False,
    )

    for batch_idx, profile_id in enumerate(test_profiles):
        profile_df = (
            grouped.get_group(profile_id)
            .sort_values(cfg.TS_COL)
            .reset_index(drop=True)
        )

        n_pairs = min(
            len(profile_df) - 1,
            predicted_c.shape[1],
        )

        if n_pairs <= 0:
            continue

        valid = mask_np[batch_idx, :n_pairs]

        # Predictions generated from row k are aligned with measured row k+1.
        block = (
            profile_df.iloc[1 : n_pairs + 1]
            .reset_index(drop=True)
            .loc[valid]
            .copy()
        )

        pred_block = predicted_c[
            batch_idx,
            :n_pairs,
            :,
        ][valid]

        blocks.append(block)
        pred_blocks.append(pred_block)

    if not blocks:
        raise RuntimeError(
            "No valid TNN prediction pairs were produced."
        )

    result_df = pd.concat(
        blocks,
        ignore_index=True,
    )
    predicted_valid_c = np.concatenate(
        pred_blocks,
        axis=0,
    )

    return result_df, predicted_valid_c


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_config = cfg.CONFIG
    data_path = Path(cfg.DATA_PATH)

    model_path = (
        f"{cfg.DS}_{model_config}_tnn_winding_baseline.pt"
    )

    model, metadata = load_tnn(
        model_path,
        device,
    )

    input_cols = metadata["input_cols"]
    target_cols = metadata["target_cols"]
    x_scaler = metadata["x_scaler"]
    y_scaler = metadata["y_scaler"]
    dt_s = metadata["dt_s"]
    window_steps = metadata["window_steps"]

    print("Model loaded successfully")
    print("Dataset:", metadata.get("dataset", cfg.DS))
    print("Configuration:", metadata.get("experiment_config", cfg.CONFIG))
    print("Input columns:", input_cols)
    print("Target columns:", target_cols)
    print("Sampling interval:", dt_s)

    if USE_ANOMALY_FILE:
        test_path = (
            data_path
            / "train_test_split"
            / "with_anomalies"
            / f"{cfg.DS}_generator_test_w_anomalies.csv"
        )
    else:
        test_path = (
            data_path
            / "train_test_split"
            / f"{cfg.DS}_generator_test.csv"
        )

    (
        test_data,
        actual_df,
        test_profiles,
        test_tensor,
        test_mask,
    ) = load_and_prepare_test_data(
        csv_path=test_path,
        input_cols=input_cols,
        target_cols=target_cols,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        checkpoint_dt_s=dt_s,
        window_steps=window_steps,
        device=device,
    )

    predicted_c, pair_mask = predict_tnn(
        model=model,
        test_tensor=test_tensor,
        test_mask=test_mask,
        input_cols=input_cols,
        target_cols=target_cols,
        y_scaler=y_scaler,
    )

    result_df, predicted_valid_c = flatten_aligned_results(
        actual_df=actual_df,
        test_profiles=test_profiles,
        predicted_c=predicted_c,
        pair_mask=pair_mask,
        target_cols=target_cols,
    )

    actual_c = result_df[target_cols].to_numpy(
        dtype=float
    )

    residuals_c = actual_c - predicted_valid_c
    missing_measurements = ~np.isfinite(actual_c)

    anomalies = (
        np.abs(residuals_c) > ANOMALY_THRESHOLD
    ) | missing_measurements

    anomaly_cols = []

    for j, target_col in enumerate(target_cols):
        result_df[f"{target_col}_predicted"] = (
            predicted_valid_c[:, j]
        )
        result_df[f"{target_col}_residual"] = (
            residuals_c[:, j]
        )
        result_df[f"{target_col}_missing"] = (
            missing_measurements[:, j]
        )
        result_df[f"{target_col}_anomaly"] = (
            anomalies[:, j]
        )

        anomaly_cols.append(
            f"{target_col}_anomaly"
        )

    result_df["any_anomaly"] = (
        result_df[anomaly_cols].any(axis=1)
    )

    result_path = (
        data_path
        / "results"
        / f"{cfg.DS}_{model_config}_tnn_anomaly_results.csv"
    )
    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result_df.to_csv(
        result_path,
        index=False,
    )

    print(f"Results saved to: {result_path}")

    plot_residuals_inference(
        data=result_df,
        residual=residuals_c,
        target_cols=target_cols,
        anomaly_threshold=ANOMALY_THRESHOLD,
        show_threshold=False,
    )

    plot_actual_vs_predicted(
        actual_df=result_df,
        predicted=predicted_valid_c,
        target_cols=target_cols,
        anomaly_threshold=ANOMALY_THRESHOLD,
    )

    plot_predicted_vs_actual_inference(
        actual_df=result_df,
        predicted=predicted_valid_c,
        target_cols=target_cols,
    )

    timestamps = pd.to_datetime(
        result_df[cfg.TS_COL],
        errors="coerce",
        utc=True,
    ).dropna()

    if len(timestamps) >= 2:
        hours = (
            timestamps.iloc[-1] - timestamps.iloc[0]
        ).total_seconds() / 3600.0

        print(
            f"Prediction horizon: {hours:.2f} hours / "
            f"{hours / 24.0:.2f} days"
        )

    print(
        "Detected anomalous observations:",
        int(result_df["any_anomaly"].sum()),
    )


if __name__ == "__main__":
    main()
