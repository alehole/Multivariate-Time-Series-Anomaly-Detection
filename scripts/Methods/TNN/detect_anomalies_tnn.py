from __future__ import annotations
from pathlib import Path
import torch
from torch import Tensor
from src.models.TNN.tnn_model import build_model
from src.models.TNN.train_tnn_model import predict_test_set
from src.visualization.prediction_plots import plot_predicted_vs_actual_inference, plot_actual_vs_predicted
from src.visualization.residual_plots import plot_residuals_inference
from src.models.fault_injection import *
from src.preprocess.feature_engineering import ts_cols, feature_processing
from src.config import paths

from src.models.model_utils import (
    get_prediction_window,
)
import numpy as np
import pandas as pd
def load_model(
        path: str,
        device: str = "cpu"
):
    device = torch.device(device)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    # Use saved hyperparameters if available
    if "best_params" in checkpoint:
        best_params = checkpoint["best_params"]
        print("Loaded hyperparameters:", best_params)

        n_neurons = best_params["n_neurons"]
    else:
        print("No best_params found, using fallback n_neurons")
        n_neurons = checkpoint["n_neurons"]

    model = build_model(
        dt_s=checkpoint["dt_s"],
        input_cols=checkpoint["input_cols"],
        target_cols=checkpoint["target_cols"],
        temperature_cols=checkpoint["temperature_cols"],
        device=device,
        n_neurons=n_neurons,
        cooling_columns=checkpoint["cooling_columns"],
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return (
        model,
        checkpoint["x_scaler"],
        checkpoint["y_scaler"],
        checkpoint["input_cols"],
        checkpoint["target_cols"],
        checkpoint["temperature_cols"],
        checkpoint["dt_s"],
        checkpoint.get("best_params", None),
    )


def tensorize_inference(
    data: pd.DataFrame,
    input_cols: list[str],
    target_cols: list[str],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """
    Convert a single dataframe into one padded inference tensor.

    Returns
    -------
    test_tensor : Tensor
        Shape (1, T, F)
    test_mask : Tensor
        Shape (1, T)

    1=B meaning batch size = 1
    """

    cols = input_cols + target_cols
    df = data.loc[:, cols].copy()

    arr = (
        df.apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float32)
    )

    # one sequence only -> batch size = 1
    tensor = arr[None, :, :]   # (1, T, F)

    # valid timestep = row is not all NaN
    mask = ~np.all(np.isnan(tensor), axis=2)   # (1, T)

    tensor = np.nan_to_num(tensor, nan=0.0).astype(np.float32)

    return (
        torch.from_numpy(tensor).to(device),
        torch.from_numpy(mask).to(device),
    )

def main():
    ANOMALY_THRESHOLD = 5
    PREDICTION_HOURS_START = 0  # start prediction X hours into dataset
    PREDICTION_HOURS = 120  # predict for next X hours
    # --------------------------------
    # 1 Load trained model
    # --------------------------------
    path = paths.file_TNN_MODEL_cyl_all

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model, x_scaler, y_scaler, input_cols, target_cols, temperature_cols, dt_s, best_params = load_model(str(path))

    print()
    print("Model loaded successfully")
    print()

    print("Input columns:")
    print(input_cols)

    print()
    print("Target columns:")
    print(target_cols)

    print()
    print("Temperature nodes:", len(temperature_cols))
    print("Sample time:", dt_s)

    # --------------------------------
    # 2 Load new generator data( Actually the test set )
    # --------------------------------
    data = pd.read_csv("test_set.csv")

    # --------------------------------
    # Inject Drift Fault on e.g a winding temperature
    # --------------------------------
    inj_drift_fault = False
    inj_sensor_dropout = False
    inj_noise_fault = False
    inj_bias_fault = False
    inj_stuck_sensor = False

    if inj_drift_fault:
        data = inject_drift_fault(
            df=data,
            col="AE PORT GEN.V-WINDING TEMP.",
            start_idx=200,
            end_idx=1000,
            final_drift=3.0,
        )

    if inj_sensor_dropout:
        data = inject_sensor_dropout(
            df=data,
            col="AE PORT GEN.U-WINDING TEMP.",
            start_idx=100,
            end_idx=600)


    if inj_noise_fault:
        data = inject_noise_fault(
            df=data,
            col="AE PORT GEN.U-WINDING TEMP.",
            start_idx=100,
            end_idx=600,
            noise_std=1.0,
        )

    if inj_bias_fault:
        data = inject_bias_fault(
            data,
            "AE PORT GEN.W-WINDING TEMP.",
            start_idx=200,
            bias=3.0,
        )
    if inj_stuck_sensor:
        data = inject_stuck_sensor(
        data,
            "AE PORT GEN.U-WINDING TEMP.",
            start_idx=800,
        )

    # --------------------------------
    # 3 Same preprocessing as training
    # --------------------------------
    data, data_dt_s = ts_cols(data, "Created")
    data = feature_processing(data)
    # --------------------------------
    # 4 Ensure numeric columns
    # --------------------------------
    data[input_cols] = data[input_cols].apply(pd.to_numeric, errors="coerce")
    data[target_cols] = data[target_cols].apply(pd.to_numeric, errors="coerce")

    # Fill inputs so model remains stable
    data[input_cols] = data[input_cols].ffill().bfill()
    # Do NOT fill targets
    data[target_cols] = data[target_cols].apply(pd.to_numeric, errors="coerce")

    # --------------------------------
    # Limit prediction window
    # --------------------------------
    data["Created"] = pd.to_datetime(data["Created"], errors="coerce", utc=True)
    data = data.dropna(subset=["Created"]).sort_values("Created").reset_index(drop=True)

    data = get_prediction_window(
        data=data,
        start_h=PREDICTION_HOURS_START,
        horizon_h=PREDICTION_HOURS,
    )
    # Keep true temperatures for only this window
    actual_df = data[["Created"] + target_cols].copy()

    # --------------------------------
    # 5 Apply saved scalers
    # --------------------------------
    data[input_cols] = x_scaler.transform(data[input_cols])
    data[target_cols] = y_scaler.transform(data[target_cols])


    # --------------------------------
    # 6 Convert to tensor
    # --------------------------------
    test_tensor, test_mask = tensorize_inference(
        data=data,
        input_cols=input_cols,
        target_cols=target_cols,
        device=torch.device("cpu"),
    )
    # --------------------------------
    # 7 Predict temperatures
    # --------------------------------
    pred_c = predict_test_set(
        model,
        test_tensor,
        input_cols,
        target_cols,
        y_scaler
    )
    # --------------------------------
    # 8 Residuals
    # --------------------------------
    actual_c = actual_df[target_cols].to_numpy()
    predicted_c = pred_c[0, :len(actual_c), :]
    residual = actual_c - predicted_c

    # --------------------------------
    # 9 Build result dataframe
    # --------------------------------
    result_df = actual_df.copy()
    for j, col in enumerate(target_cols):
        result_df[f"{col}_predicted"] = predicted_c[:, j]
        result_df[f"{col}_residual"] = residual[:, j]
        result_df[f"{col}_anomaly"] = np.abs(residual[:, j]) > ANOMALY_THRESHOLD

    # --------------------------------
    # 10 Save results
    # --------------------------------
    result_df.to_csv(paths.file_TNN_ANOMALY_RESULTS_xxxxx, index=False)
    # --------------------------------
    # 11 Residual Plot
    # --------------------------------
    plot_residuals_inference(
        data=result_df,
        residual=residual,
        target_cols=target_cols,
        anomaly_threshold=ANOMALY_THRESHOLD,
    )
    # --------------------------------
    # 12 Actual vs Predicted Plot
    # --------------------------------
    plot_actual_vs_predicted(
        actual_df=actual_df,
        predicted=predicted_c,
        target_cols=target_cols,
        anomaly_threshold=ANOMALY_THRESHOLD,
    )

    # --------------------------------
    # 13 Actual vs Predicted Scatter Plot
    # --------------------------------
    plot_predicted_vs_actual_inference(
        actual_df=actual_df,
        predicted=predicted_c,
        target_cols=target_cols,
    )
    # --------------------------------
    # Prediction duration
    # --------------------------------
    t0 = pd.to_datetime(actual_df["Created"].iloc[0])
    t1 = pd.to_datetime(actual_df["Created"].iloc[-1])

    duration = t1 - t0
    hours = duration.total_seconds() / 3600
    days = hours / 24
    print()
    print(f"Prediction horizon: {hours:.2f} hours/{days:.2f} days")
    print(f"Start time: {t0}")
    print(f"End time:   {t1}")
    print()

if __name__ == "__main__":
    main()