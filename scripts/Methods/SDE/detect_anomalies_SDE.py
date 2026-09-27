from pathlib import Path
import pickle
from model import simulate_model
from plotting import plot_full_ekf_comparison
import numpy as np
import config as cfg

from misc.data import load_data
from ekf import run_ekf

from Visualization.prediction_plots import (
    plot_actual_vs_predicted,
    plot_predicted_vs_actual_inference,
)

from Visualization.residual_plots import (
    plot_residuals_inference,
)


def load_sde_checkpoint(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"SDE checkpoint not found: {path}"
        )

    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    return checkpoint


def main():

    # -----------------------------------------------------
    # 1. Select fitted SDE and anomaly dataset
    # -----------------------------------------------------
    model_path = Path(cfg.DATA_PATH)/"models"/"ds1_2state_B_sde.pkl"
    data_path = Path(cfg.DATA_PATH)
    test_path = data_path / "train_test_split/ds1_generator_test.csv"

    # -----------------------------------------------------
    # 2. Load fitted SDE
    # -----------------------------------------------------

    checkpoint = load_sde_checkpoint(model_path)
    theta_hat = checkpoint["theta_hat"]
    state_cols = checkpoint["state_cols"]
    meas_cols = checkpoint["meas_cols"]
    input_cols = checkpoint["input_cols"]
    Q = checkpoint["Q"]
    R = checkpoint["R"]
    C = checkpoint["C"]

    rename_map = checkpoint["rename_map"]
    sensor_cols = checkpoint["sensor_cols"]

    #threshold = checkpoint["anomaly_threshold"]
    threshold = 3.0

    print("SDE model loaded successfully")
    print("Model:", checkpoint["model_option"])
    print("Parameters:", theta_hat)
    print("Measurements:", meas_cols)
    print("Inputs:", input_cols)

    # -----------------------------------------------------
    # 3. Load anomaly-injected test data
    # -----------------------------------------------------

    df_test = load_data(test_path, cfg.TS_COL, sensor_cols, rename_map)
    print("Test observations:", len(df_test))

    # -----------------------------------------------------
    # 4. Run fitted EKF
    #
    # -----------------------------------------------------

    (
        x_hat,
        P_cov,
        innovations,
        nis,
        x_pred,
        likelihood,
    ) = run_ekf(
        df=df_test,
        theta=theta_hat,
        C=C,
        Q=Q,
        R=R,
        state_cols=state_cols,
        meas_cols=meas_cols,
        input_cols=input_cols,
        model_option=checkpoint["model_option"],
    )

    # -----------------------------------------------------
    # 5. Convert prior state prediction to measurements
    # -----------------------------------------------------

    predicted = np.asarray(x_pred, dtype=float) @ np.asarray(C, dtype=float).T

    actual = df_test[meas_cols].to_numpy(dtype=float)

    # -----------------------------------------------------
    # 6. Prior residual
    # -----------------------------------------------------

    residuals = actual - predicted

    # First prior is only initialization
    residuals[0] = np.nan

    # -----------------------------------------------------
    # 7. Detect anomalies
    # -----------------------------------------------------

    anomalies = np.abs(residuals) > threshold

    result_df = df_test[[cfg.TS_COL, *meas_cols]].copy()

    anomaly_cols = []

    for j, col in enumerate(meas_cols):

        result_df[f"{col}_predicted"] = predicted[:, j]
        result_df[f"{col}_residual"] = residuals[:, j]
        result_df[f"{col}_anomaly"] = anomalies[:, j]

        anomaly_cols.append(f"{col}_anomaly")

    result_df["any_anomaly"] = result_df[anomaly_cols].any(axis=1)

    # -----------------------------------------------------
    # 8. Results
    # -----------------------------------------------------

    n_anomalies = int(result_df["any_anomaly"].sum())

    print()
    print(
        f"Residual threshold: "
        f"{threshold:.2f} °C"
    )

    print(
        f"Detected anomalies: "
        f"{n_anomalies}"
    )

    print(
        f"Anomaly percentage: "
        f"{100*n_anomalies/len(result_df):.3f}%"
    )

    # -----------------------------------------------------
    # 9. Save
    # -----------------------------------------------------

    result_path = (
        Path(cfg.DATA_PATH)
        / "results"
        / "ds1_1state_sde_anomaly_results.csv"
    )

    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        result_path,
        index=False,
    )

    # -----------------------------------------------------
    # 10. Plots
    # -----------------------------------------------------
    target_cols = ["T1"]
    target_idx = [meas_cols.index("T1")]
    predicted_T1 = predicted[:, target_idx]
    residuals_T1 = residuals[:, target_idx]

    actual_df_T1 = result_df[
        [cfg.TS_COL, "T1"]
    ].copy()

    plot_residuals_inference(
        data=result_df,
        residual=residuals_T1,
        target_cols=["T1"],
        anomaly_threshold=threshold,
        show_threshold=True,
    )

    plot_actual_vs_predicted(
        actual_df=actual_df_T1,
        predicted=predicted_T1,
        target_cols=["T1"],
        anomaly_threshold=threshold,
    )

    plot_predicted_vs_actual_inference(
        actual_df=actual_df_T1,
        predicted=predicted_T1,
        target_cols=["T1"],
    )

    # Open-loop simulation
    x_sim_test = simulate_model(
        df_test,
        theta_hat,
    )

    plot_full_ekf_comparison(
        df_test,
        x_sim_test,
        x_pred,
        x_hat,
        title="Testing: Model vs EKF",
        target_cols=["T1"],
    )

if __name__ == "__main__":
    main()