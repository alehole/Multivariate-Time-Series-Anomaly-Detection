from pathlib import Path
import numpy as np
import pandas as pd
import torch

from Visualization.prediction_plots import plot_predicted_vs_actual_inference, plot_actual_vs_predicted
from Visualization.residual_plots import plot_residuals_inference
import config as cfg
from Methods.Blackbox.TCN.TCN import TCNBaseline
from Methods.Blackbox.profile_dataset import (
    create_profiles,
    tensorize_profiles,
)
from Methods.Blackbox.common_BB_scripts import predict_sequence_model, load_sequence_model
from scripts.misc.feature_engineering import ts_cols
import Methods.Blackbox.experiment_configs as method_cfg
def load_and_prepare_test_data(
    csv_path: str | Path,
    input_cols: list[str],
    target_cols: list[str],
    x_scaler,
    y_scaler,
    checkpoint_dt_s: float,
    device: torch.device,
):
    """
    Load and prepare a fault-injected test set using the same
    preprocessing objects and profile configuration used during training.
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Test dataset not found: {csv_path}")

    # -----------------------------------------------------
    # 1. Load data and process timestamps
    # -----------------------------------------------------
    data = pd.read_csv(csv_path)
    data, data_dt_s = ts_cols(data, cfg.TS_COL)

    if not np.isclose(data_dt_s, checkpoint_dt_s, rtol=0.05):
        print(
            "Warning: sampling interval differs from training: "
            f"test={data_dt_s:.2f} s, "
            f"training={checkpoint_dt_s:.2f} s"
        )

    # Keep optional synthetic-fault labels if they exist.
    label_cols = [
        col
        for col in ["fault_label", "fault_type", "fault_id"]
        if col in data.columns
    ]

    required_raw_cols = [cfg.TS_COL, *cfg.SENSOR_COLS]

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
        dict.fromkeys([*required_raw_cols, *label_cols])
    )

    data = data[keep_cols].copy()

    # Apply the same column names used during training.
    data = data.rename(columns=cfg.RENAME_MAP)

    required_model_cols = [*input_cols, *target_cols]

    missing_model_cols = [
        col
        for col in required_model_cols
        if col not in data.columns
    ]

    if missing_model_cols:
        raise KeyError(
            "Prepared test data is missing checkpoint columns: "
            f"{missing_model_cols}"
        )

    # -----------------------------------------------------
    # 2. Ensure numeric model variables
    # -----------------------------------------------------
    data[input_cols] = data[input_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    data[target_cols] = data[target_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # -----------------------------------------------------
    # 3. Create profiles before scaling
    # -----------------------------------------------------
    data, test_profiles = create_profiles(
        data,
        ts_col=cfg.TS_COL,
        window_steps=len(data),
        dt_s=data_dt_s,
    )

    # Preserve measurements in physical units for residuals.
    actual_cols = [cfg.TS_COL, *target_cols, *label_cols]

    actual_df = data[
        [col for col in actual_cols if col in data.columns]
    ].copy()

    # -----------------------------------------------------
    # 4. Apply training-fitted scalers
    # -----------------------------------------------------
    data[input_cols] = x_scaler.transform(
        data[input_cols]
    )

    data[target_cols] = y_scaler.transform(
        data[target_cols]
    )

    # -----------------------------------------------------
    # 5. Convert profiles to tensors
    # -----------------------------------------------------
    x_test, y_test, mask_test = tensorize_profiles(
        data,
        test_profiles,
        input_cols,
        target_cols,
        device=device,
    )

    return (
        data,
        actual_df,
        test_profiles,
        x_test,
        y_test,
        mask_test,
    )

def load_trained_tcn(
    model_path: str | Path,
    device: torch.device,
):
    model_registry = {
        "TCN": TCNBaseline,
        "TCNBaseline": TCNBaseline,
    }

    return load_sequence_model(
        path=model_path,
        model_registry=model_registry,
        device=device,
    )

def main():
    CALC_THRESHOLDS = False
    ANOMALY_THRESHOLDS = [4.270, 3.291, 3.838]

    model_config = cfg.CONFIG
    # -----------------------------------------------------
    # 1. Load pretrained model
    # -----------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, metadata = load_trained_tcn(model_path=f"{cfg.DS}_{model_config}_tcn_winding_baseline.pt",  device=device)
    #model, metadata = load_trained_tcn(model_path=f"ds2_{model_config}_tcn_winding_baseline.pt", device=device)

    x_scaler = metadata["x_scaler"]
    y_scaler = metadata["y_scaler"]
    input_cols = metadata["input_cols"]
    target_cols = metadata["target_cols"]
    dt_s = metadata["dt_s"]

    print("Model loaded successfully")
    print("Input columns:", input_cols)
    print("Target columns:", target_cols)
    print("Sampling interval:", dt_s)

    # -----------------------------------------------------
    # 2. Load and prepare anomaly test dataset
    # -----------------------------------------------------
    data_path = Path(cfg.DATA_PATH)

    #test_path = data_path / "train_test_split" / "with_anomalies" / "ds1_F4_T5_test.csv"

    test_path = data_path/ "train_test_split"/ f"{cfg.DS}_generator_test.csv"
    (
        test_data,
        actual_df,
        test_profiles,
        x_test,
        y_test,
        mask_test,
    ) = load_and_prepare_test_data(
        csv_path=test_path,
        input_cols=input_cols,
        target_cols=target_cols,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        checkpoint_dt_s=dt_s,
        device=device,
    )

    print()
    print("Anomaly test set prepared")
    print("Input tensor shape:", x_test.shape)
    print("Target tensor shape:", y_test.shape)
    print("Mask shape:", mask_test.shape)

    print("TCN model loaded successfully")
    print("Device:", device)
    print("Model type:", metadata["model_type"])
    print("Model class:", metadata["model_class"])
    print("Input columns:", input_cols)
    print("Target columns:", target_cols)
    print("Sampling interval:", dt_s)

    # -----------------------------------------------------
    # 3. Predict winding temperatures
    # -----------------------------------------------------
    predicted_c = predict_sequence_model(
        model=model,
        x=x_test,
        y_scaler=y_scaler,
    )

    print("Profile prediction shape:", predicted_c.shape)

    # -----------------------------------------------------
    # 4. Flatten valid profile predictions
    # -----------------------------------------------------
    mask = mask_test.detach().cpu().numpy().astype(bool)
    predicted_valid_c = predicted_c[mask]

    actual_c = actual_df[target_cols].to_numpy(
        dtype=float
    )

    if predicted_valid_c.shape != actual_c.shape:
        raise RuntimeError(
            "Prediction and measurement shapes do not match: "
            f"predictions={predicted_valid_c.shape}, "
            f"measurements={actual_c.shape}."
        )

    # -----------------------------------------------------
    # 5. Calculate residuals and anomaly thresholds
    # -----------------------------------------------------
    residuals_c = actual_c - predicted_valid_c

    missing_measurements = ~np.isfinite(actual_c)


    if CALC_THRESHOLDS:
        # One 99.5th percentile threshold for each output
        ANOMALY_THRESHOLDS = np.nanquantile(
            np.abs(residuals_c),
            0.995,
            axis=0,
        )

    for target, threshold in zip(target_cols, ANOMALY_THRESHOLDS):
        print(f"{target} anomaly threshold: {threshold:.3f} °C")

    anomalies = (
                        np.abs(residuals_c) > ANOMALY_THRESHOLDS
                ) | missing_measurements
    # -----------------------------------------------------
    # 6. Build result dataframe
    # -----------------------------------------------------
    result_df = actual_df.copy()
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

    # -----------------------------------------------------
    # 7. Save results
    # -----------------------------------------------------
    result_path = (
        data_path
        / "results"
        / "tcn_anomaly_results.csv"
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

    # -----------------------------------------------------
    # 8. Plot residuals
    # -----------------------------------------------------
    plot_residuals_inference(
        data=result_df,
        residual=residuals_c,
        target_cols=target_cols,
        anomaly_threshold=ANOMALY_THRESHOLDS,
        show_threshold=False,
        #plot_targets=["T1", "T2", "T3"],
        plot_targets=["T1"],
    )

    # -----------------------------------------------------
    # 9. Plot actual versus predicted
    # -----------------------------------------------------
    plot_actual_vs_predicted(
        actual_df=actual_df,
        predicted=predicted_valid_c,
        target_cols=target_cols,
        anomaly_threshold=ANOMALY_THRESHOLDS,
    )

    # -----------------------------------------------------
    # 10. Plot actual versus predicted scatter
    # -----------------------------------------------------
    plot_predicted_vs_actual_inference(
        actual_df=actual_df,
        predicted=predicted_valid_c,
        target_cols=target_cols,
    )

    # -----------------------------------------------------
    # 11. Prediction duration
    # -----------------------------------------------------
    timestamps = pd.to_datetime(
        actual_df[cfg.TS_COL],
        errors="coerce",
        utc=True,
    ).dropna()

    if len(timestamps) >= 2:
        t0 = timestamps.iloc[0]
        t1 = timestamps.iloc[-1]

        hours = (
            t1 - t0
        ).total_seconds() / 3600.0

        print()
        print(
            f"Prediction horizon: "
            f"{hours:.2f} hours / "
            f"{hours / 24.0:.2f} days"
        )
        print(f"Start time: {t0}")
        print(f"End time:   {t1}")

    print()
    print(
        "Detected anomalous observations:",
        int(result_df["any_anomaly"].sum()),
    )

if __name__ == "__main__":
    main()