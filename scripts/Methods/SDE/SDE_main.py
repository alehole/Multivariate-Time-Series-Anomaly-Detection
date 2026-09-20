from misc.data import load_data
from pathlib import Path
import pickle
import pandas as pd
import config as cfg
import SDE_config as sde_cfg
from notify_phone import notify_phone
from parameter_estimation import estimate_parameters_mle, wilks_likelihood_ratio_test
from PL1 import run_pl1
from PL2 import run_pl2
from plotting import(
    plot_simulated_vs_actual,
    print_metrics,
    plot_residuals_vs_time,
    plot_results,
    plot_innovations,
    plot_NIS,
    plot_full_ekf_comparison
)
from Visualization.prediction_plots import (
    plot_predicted_vs_actual_inference,
    plot_actual_vs_predicted,
)

from Visualization.residual_plots import (
    plot_residuals_inference,
)
from model import simulate_model
from ekf import run_ekf
import numpy as np


def save_sde_checkpoint(
    path,
    theta_hat,
    df_train,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = pd.to_datetime(
        df_train[cfg.TS_COL],
        errors="coerce",
        utc=True,
    )

    dt_s = timestamps.diff().dt.total_seconds().median()

    checkpoint = {
        "model_option": sde_cfg.MODEL_OPTION,

        "theta_hat": theta_hat,

        "state_cols": sde_cfg.STATE_COLS,
        "meas_cols": sde_cfg.MEAS_COLS,
        "input_cols": sde_cfg.INPUT_COLS,

        "parameter_names": sde_cfg.PARAMETER_NAMES,

        "Q": sde_cfg.Q,
        "R": sde_cfg.R,
        "C": sde_cfg.C,

        "rename_map": sde_cfg.RENAME_MAP,
        "sensor_cols": sde_cfg.SENSOR_COLS,

        "dt_s": dt_s,

        "anomaly_threshold":
            sde_cfg.RESIDUAL_ANOMALY_THRESHOLD,
    }

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"SDE checkpoint saved to: {path}")


def detect_residual_anomalies(
    df,
    x_pred,
    threshold_c,
    burn_in=1,
):
    """
    Detect anomalies using the absolute EKF prior residual.

    An observation is flagged as anomalous when

        abs(measurement - prediction) > threshold_c

    Missing measurements are also flagged as anomalies.

    Parameters
    ----------
    df:
        Dataset containing the measured output variables.
    x_pred:
        EKF prior state estimates with shape (N, n_states).
    threshold_c:
        Absolute residual threshold in degrees Celsius.
    burn_in:
        Number of initial predictions excluded from anomaly detection.

    Returns
    -------
    result_df:
        Dataframe containing measurements, predictions, residuals,
        missing-value indicators, and anomaly flags.
    """

    x_pred = np.asarray(x_pred, dtype=float)
    C = np.asarray(sde_cfg.C, dtype=float)

    # ------------------------------------------------------------
    # Convert predicted states to predicted measurements
    #
    # x_pred has shape:
    #     (N, n_states)
    #
    # C has shape:
    #     (n_measurements, n_states)
    #
    # Therefore, predicted_measurements has shape:
    #     (N, n_measurements)
    # ------------------------------------------------------------
    predicted_measurements = x_pred @ C.T

    actual_measurements = df[
        sde_cfg.MEAS_COLS
    ].to_numpy(dtype=float)

    if predicted_measurements.shape != actual_measurements.shape:
        raise ValueError(
            "Predicted and measured output shapes do not match: "
            f"predicted={predicted_measurements.shape}, "
            f"actual={actual_measurements.shape}."
        )

    # Prediction residual:
    #
    #     residual = measurement - prior prediction
    residuals = (
        actual_measurements
        - predicted_measurements
    )

    missing_measurements = ~np.isfinite(
        actual_measurements
    )

    # Flag large residuals or missing measurements.
    anomalies = (
        np.abs(residuals) > threshold_c
    ) | missing_measurements

    # The first EKF prior is normally initialized rather than
    # produced by a genuine one-step-ahead prediction.
    if burn_in > 0:
        residuals[:burn_in] = np.nan
        anomalies[:burn_in] = False

    # ------------------------------------------------------------
    # Build results dataframe
    # ------------------------------------------------------------
    result_columns = [
        col
        for col in [
            cfg.TS_COL,
            *sde_cfg.MEAS_COLS,
        ]
        if col in df.columns
    ]

    result_df = df[result_columns].copy()

    anomaly_columns = []

    for j, measurement_col in enumerate(
            sde_cfg.MEAS_COLS
    ):
        result_df[
            f"{measurement_col}_predicted"
        ] = predicted_measurements[:, j]

        result_df[
            f"{measurement_col}_residual"
        ] = residuals[:, j]

        result_df[
            f"{measurement_col}_missing"
        ] = missing_measurements[:, j]

        # Use the same naming convention as the TCN detector.
        anomaly_col = f"{measurement_col}_anomaly"

        result_df[anomaly_col] = anomalies[:, j]
        anomaly_columns.append(anomaly_col)

    result_df["any_anomaly"] = (
        result_df[anomaly_columns].any(axis=1)
    )

    return result_df

def print_all_metrics(
    df_train,
    df_test,
    x_sim_train,
    x_sim_test,
    x_pred_train,
    x_pred_test,
    x_hat_train,
    x_hat_test,
):
    """
    Report performance for the open-loop model, EKF prior prediction,
    and EKF posterior state estimate on training and test data.
    """


    # ------------------------------------------------------------
    # Open-loop grey-box model
    #
    # The state is propagated using only the model and measured
    # inputs. No measurement corrections are applied.
    # ------------------------------------------------------------
    print_metrics(df_train, x_sim_train, "Training Open-loop Simulation")
    print_metrics(df_test, x_sim_test, "Testing Open-loop Simulation")

    # ------------------------------------------------------------
    # EKF prior state estimate (prediction)
    # ------------------------------------------------------------
    print_metrics(df_train, x_pred_train, "Training EKF Prior Estimate")
    print_metrics(df_test, x_pred_test, "Testing EKF Prior Estimate")

    # ------------------------------------------------------------
    # EKF posterior state estimate (measurement update)
    # ------------------------------------------------------------
    print_metrics(df_train, x_hat_train, "Training EKF Posterior Estimate")
    print_metrics(df_test, x_hat_test, "Testing EKF Posterior Estimate")


def report_nis(
    name: str,
    nis: np.ndarray,
    dof: int,
    likelihood_cost: float,
    burn_in: int = 1,
) -> None:
    """
    Report the mean normalized innovation squared (NIS) and
    the accumulated EKF likelihood cost.

    For a statistically consistent filter, the expected mean NIS
    is approximately equal to the number of measured variables.
    """

    nis = np.asarray(nis, dtype=float)

    if burn_in < 0 or burn_in >= len(nis):
        raise ValueError(
            f"burn_in must be between 0 and {len(nis) - 1}."
        )

    # Ignore the initial samples and any undefined NIS values.
    nis_valid = nis[burn_in:]
    nis_valid = nis_valid[np.isfinite(nis_valid)]

    if len(nis_valid) == 0:
        raise ValueError(
            f"No valid NIS values are available for {name}."
        )

    mean_nis = nis_valid.mean()
    nis_ratio = mean_nis / dof

    print(f"\nNIS diagnostics ({name}):")
    print(f"Mean NIS:          {mean_nis:.3f}")
    print(f"Expected mean:     {dof}")
    print(f"NIS/DOF ratio:     {nis_ratio:.2f}")
    print(f"Likelihood cost:   {likelihood_cost:.3f}")



def load_datasets():
    data_path = Path(cfg.DATA_PATH)

    if sde_cfg.RUN_TOY_CHECK:
        train_path = data_path / "toy_sim/toy_generator_train.csv"
        test_path = data_path / "toy_sim/toy_generator_test.csv"
    else:
        train_path = data_path / "train_test_split/ds1_generator_train.csv"
        test_path = data_path / "train_test_split/ds1_generator_test.csv"

    df_train = load_data(
        train_path,
        cfg.TS_COL,
        sde_cfg.SENSOR_COLS,
        sde_cfg.RENAME_MAP,
    )

    df_test = load_data(
        test_path,
        cfg.TS_COL,
        sde_cfg.SENSOR_COLS,
        sde_cfg.RENAME_MAP,
    )

    return df_train, df_test

def fit_model(df_train):
    theta_hat, result = estimate_parameters_mle(df_train)

    if sde_cfg.RUN_TOY_CHECK:
        # For the toy, compare against known truth
        print("\nRecovery check (est / true):")
        for name, est, true in zip(sde_cfg.PARAMETER_NAMES, theta_hat, sde_cfg.TOY_THETA_TRUE):
            print(f"  {name}: {est:.5g} vs {true:.5g}  ({100 * est / true:.3f}%)")

    print("\nEstimated parameters:")
    for name, value in zip(
        sde_cfg.PARAMETER_NAMES,
        theta_hat,
    ):
        print(f"{name}: {value:.6g}")

    print("\nOptimization diagnostics:")
    print("Success:", result.success)
    print("Message:", result.message)
    print("Final likelihood cost:", result.fun)
    print("Iterations:", result.get("nit", "n/a"))
    print(
        "Function evaluations:",
        result.get("nfev", "n/a"),
    )
    return theta_hat, result

def main():

    # -----------------------------------------------------
    # Load training and testing datasets
    # -----------------------------------------------------
    df_train, df_test = load_datasets()

    print(f"Training observations: {len(df_train):,}")
    print(f"Test observations:     {len(df_test):,}")
    print(f"Selected model:        {sde_cfg.MODEL_OPTION}")

    # -----------------------------------------------------
    # Fit parameters
    # -----------------------------------------------------
    theta_hat, result = fit_model(df_train)

    # -----------------------------------------------------
    # Save fitted model
    # -----------------------------------------------------
    save_sde_checkpoint(
        path=Path(cfg.DATA_PATH)/"models"/f"{cfg.DS}_{sde_cfg.MODEL_OPTION}_sde.pkl",
        theta_hat=theta_hat,
        df_train=df_train,
    )

    # -----------------------------------------------------
    # Profile-likelihood analysis
    # -----------------------------------------------------
    nll_ref = result.fun
    print("Reference NLL:", nll_ref)

    ## Profile likelihood
    pl2_results = []
    profiles = None
    if sde_cfg.RUN_PL2:
        pl2_results.append(run_pl2("C1", "R1", df_train, theta_hat, nll_ref))
        pl2_results.append(run_pl2("C2", "R2", df_train, theta_hat, nll_ref))
    if sde_cfg.RUN_PL1:
        profiles = run_pl1(df_train, theta_hat, nll_ref)

    # ------------------------------------------------------------
    # Open-loop simulation (no measurement correction)
    # ------------------------------------------------------------
    x_sim_train = simulate_model(df_train, theta_hat)
    x_sim_test = simulate_model(df_test, theta_hat)

    plot_simulated_vs_actual(df_train, x_sim_train, "Training")
    plot_simulated_vs_actual(df_test, x_sim_test, "Testing")

    # -----------------------------------------------------
    # EKF evaluation
    # -----------------------------------------------------

    (
        x_hat_train,
        P_cov_train,
        innov_train,
        nis_train,
        x_pred_train,
        likelihood_train,
    ) = run_ekf(
        df=df_train,
        theta=theta_hat,
        C=sde_cfg.C,
        Q=sde_cfg.Q,
        R=sde_cfg.R,
    )

    # The NIS degrees of freedom equal the number of measurements.
    dof = len(sde_cfg.MEAS_COLS)

    report_nis(
        name="training",
        nis=nis_train,
        dof=dof,
        likelihood_cost=likelihood_train,
        burn_in=1,
    )

    # Run the same fitted model on the independent test dataset.
    (
        x_hat_test,
        P_cov_test,
        innov_test,
        nis_test,
        x_pred_test,
        likelihood_test,
    ) = run_ekf(
        df=df_test,
        theta=theta_hat,
        C=sde_cfg.C,
        Q=sde_cfg.Q,
        R=sde_cfg.R,
    )

    report_nis(
        name="test",
        nis=nis_test,
        dof=dof,
        likelihood_cost=likelihood_test,
        burn_in=1,
    )
    # ------------------------------------------------------------
    # Evaluate model performance:
    #   - Open-loop grey-box simulation
    #   - EKF prior estimate (prediction)
    #   - EKF posterior estimate (measurement update)
    # ------------------------------------------------------------
    print_all_metrics(
        df_train,
        df_test,
        x_sim_train,
        x_sim_test,
        x_pred_train,
        x_pred_test,
        x_hat_train,
        x_hat_test,
    )

    # ------------------------------------------------------------
    # Plot estimated temperatures and uncertainty
    # ------------------------------------------------------------
    plot_results(df_train, x_hat_train, P_cov_train, "Training EKF")
    plot_results(df_test, x_hat_test, P_cov_test, "Testing EKF")
    # ------------------------------------------------------------
    # Plot innovation residuals
    # ------------------------------------------------------------
    plot_innovations(df_train, innov_train, "Training innovations")
    plot_innovations(df_test, innov_test, "Testing innovations")
    # ------------------------------------------------------------
    # Plot NIS/Mahalanobis distance for anomaly detection
    # ------------------------------------------------------------
    plot_NIS(df_train, nis_train, "Training anomaly score")
    plot_NIS(df_test, nis_test, "Testing anomaly score")

    # ------------------------------------------------------------
    # Compare:
    #   - Measured temperatures
    #   - Open-loop grey-box simulation
    #   - EKF prior estimate
    #   - EKF posterior estimate
    # ------------------------------------------------------------
    plot_full_ekf_comparison(
        df_train,
        x_sim_train,
        x_pred_train,
        x_hat_train,
        title="Training: Model vs EKF"
    )

    plot_full_ekf_comparison(
        df_test,
        x_sim_test,
        x_pred_test,
        x_hat_test,
        title="Testing: Model vs EKF"
    )

    plot_residuals_vs_time(df_train, x_sim_train, df_test, x_sim_test,
                           window="6h", title="Open-loop residual")
    # ------------------------------------------------------------
    # Residual-threshold anomaly detection
    # ------------------------------------------------------------
    sde_anomaly_results = detect_residual_anomalies(
        df=df_test,
        x_pred=x_pred_test,
        threshold_c=sde_cfg.RESIDUAL_ANOMALY_THRESHOLD,
        burn_in=1,
    )

    n_anomalies = int(
        sde_anomaly_results["any_anomaly"].sum()
    )

    print()
    print(
        "Residual anomaly threshold:",
        f"{sde_cfg.RESIDUAL_ANOMALY_THRESHOLD:.2f} °C",
    )
    print(
        "Detected residual anomalies:",
        n_anomalies,
    )
    print(
        "Anomaly percentage:",
        f"{100 * n_anomalies / len(sde_anomaly_results):.3f}%",
    )

    # ------------------------------------------------------------
    # Save anomaly results
    # ------------------------------------------------------------
    data_path = Path(cfg.DATA_PATH)
    result_path = data_path/"results"/"sde_anomaly_results.csv"

    result_path.parent.mkdir(parents=True, exist_ok=True)

    sde_anomaly_results.to_csv(result_path, index=False)

    print(f"SDE anomaly results saved to: {result_path}")

    # ------------------------------------------------------------
    # Prepare EKF-prior predictions for plotting
    # ------------------------------------------------------------
    target_cols = list(sde_cfg.MEAS_COLS)

    predicted_test = (
            np.asarray(x_pred_test, dtype=float)
            @ np.asarray(sde_cfg.C, dtype=float).T
    )

    actual_df = df_test[
        [cfg.TS_COL, *target_cols]
    ].copy()

    actual_test = actual_df[
        target_cols
    ].to_numpy(dtype=float)

    residuals_test = actual_test - predicted_test

    # The first prior estimate is initialized rather than predicted.
    residuals_test[0] = np.nan

    # ------------------------------------------------------------
    # Plot residuals and detected anomalies
    # ------------------------------------------------------------
    plot_residuals_inference(
        data=sde_anomaly_results,
        residual=residuals_test,
        target_cols=target_cols,
        anomaly_threshold=sde_cfg.RESIDUAL_ANOMALY_THRESHOLD,
    )

    # ------------------------------------------------------------
    # Plot measurements and EKF-prior predictions
    # ------------------------------------------------------------
    plot_actual_vs_predicted(
        actual_df=actual_df,
        predicted=predicted_test,
        target_cols=target_cols,
        anomaly_threshold=sde_cfg.RESIDUAL_ANOMALY_THRESHOLD,
    )

    # ------------------------------------------------------------
    # Plot predicted versus measured values
    # ------------------------------------------------------------
    plot_predicted_vs_actual_inference(
        actual_df=actual_df,
        predicted=predicted_test,
        target_cols=target_cols,
    )

    if sde_cfg.RUN_WILKS:
        theta_hat_reduced = theta_hat # Placeholders
        theta_hat_full = theta_hat # placeholders
        # fit each model = minimize neg_log_lik over its own theta
        *_, nll_A = run_ekf(df_train, theta_hat_reduced, sde_cfg.C, sde_cfg.Q, sde_cfg.R)
        *_, nll_B = run_ekf(df_train, theta_hat_full, sde_cfg.C, sde_cfg.Q, sde_cfg.R)
        LR, p = wilks_likelihood_ratio_test(nll_A, nll_B, 1)
        verdict = "keep the extra parameter" if p < 0.05 else "reduced model is adequate"
        print(f"Wilks: LR={LR:.3f}, p={p:.4f} -> {verdict}")

if __name__ == "__main__":
    main()
    if sde_cfg.RUN_PL1 or sde_cfg.RUN_PL2:
        notify_phone(
            "Pipeline has finished.",
            title="EKF finished",
        )
