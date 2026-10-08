from misc.data import load_data
from pathlib import Path
import pickle
import pandas as pd
import config as cfg
import SDE_config as sde_cfg
from notify_phone import notify_phone
from parameter_estimation import estimate_parameters_mle, wilks_likelihood_ratio_test, estimate_parameters_mle_q_theta
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

from model import simulate_model
from ekf import run_ekf
import numpy as np

def save_sde_checkpoint(
    path,
    theta_hat,
    df_train,
    Q_used,
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

        "Q": Q_used,
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
    if sde_cfg.ESTIMATE_Q:
        theta_hat, Q_used, result = estimate_parameters_mle_q_theta(df_train)
    else:
        theta_hat, result = estimate_parameters_mle(df_train)
        Q_used = np.asarray(sde_cfg.Q_INIT, dtype=float).copy()

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
    return theta_hat, result, Q_used


def evaluate_ekf(
        df,
        theta_hat,
        name,
        Q_used
):
    (
        x_hat,
        P_cov,
        innovations,
        nis,
        x_pred,
        likelihood,
    ) = run_ekf(
        df=df,
        theta=theta_hat,
        C=sde_cfg.C,
        Q=Q_used,
        R=sde_cfg.R,
    )

    report_nis(
        name=name,
        nis=nis,
        dof=len(sde_cfg.MEAS_COLS),
        likelihood_cost=likelihood,
        burn_in=1,
    )

    return {
        "x_hat": x_hat,
        "P_cov": P_cov,
        "innovations": innovations,
        "nis": nis,
        "x_pred": x_pred,
        "likelihood": likelihood,
    }


def main():

    # -----------------------------------------------------
    # Load training and testing datasets
    # -----------------------------------------------------
    df_train, df_test = load_datasets()

    print(f"Training observations: {len(df_train):,}")
    print(f"Test observations:     {len(df_test):,}")
    print(f"Selected model:        {sde_cfg.MODEL_OPTION}")

    # -----------------------------------------------------
    # Fit parameters using maximum likelihood
    # -----------------------------------------------------
    theta_hat, result , Q_used = fit_model(df_train)

    # -----------------------------------------------------
    # Save fitted model
    # -----------------------------------------------------
    save_sde_checkpoint(
        path=Path(cfg.DATA_PATH)/"models"/f"{cfg.DS}_{sde_cfg.MODEL_OPTION}_sde.pkl",
        theta_hat=theta_hat,
        df_train=df_train,
        Q_used=Q_used
    )

    # -----------------------------------------------------
    # Profile-likelihood analysis
    # -----------------------------------------------------
    if sde_cfg.RUN_PL1 or sde_cfg.RUN_PL2:
        nll_ref = result.fun
        print("Reference NLL:", nll_ref)

        if sde_cfg.RUN_PL1:
            run_pl1(
                df_train,
                theta_hat,
                nll_ref,
                Q_used
            )

        if sde_cfg.RUN_PL2:
            run_pl2(
                "C1",
                "R1",
                df_train,
                theta_hat,
                nll_ref,
                Q_used
            )
            run_pl2(
                "C2",
                "R2",
                df_train,
                theta_hat,
                nll_ref,
                Q_used
            )

    # ------------------------------------------------------------
    # Open-loop simulation (no measurement correction)
    # ------------------------------------------------------------
    x_sim_train = simulate_model(
        df_train,
        theta_hat
    )
    x_sim_test = simulate_model(
        df_test,
        theta_hat
    )

    plot_simulated_vs_actual(
        df_train,
        x_sim_train,
        "Training")

    plot_simulated_vs_actual(
        df_test,
        x_sim_test,
        "Testing"
    )

    # -----------------------------------------------------
    # EKF evaluation
    # -----------------------------------------------------

    train_ekf = evaluate_ekf(
        df_train,
        theta_hat,
        "training",
        Q_used,
    )

    test_ekf = evaluate_ekf(
        df_test,
        theta_hat,
        "test",
        Q_used,
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
        train_ekf["x_pred"],
        test_ekf["x_pred"],
        train_ekf["x_hat"],
        test_ekf["x_hat"],
    )

    plot_results(
        df_train,
        train_ekf["x_hat"],
        train_ekf["P_cov"],
        "Training EKF",
    )

    plot_results(
        df_test,
        test_ekf["x_hat"],
        test_ekf["P_cov"],
        "Testing EKF",
    )

    plot_innovations(
        df_train,
        train_ekf["innovations"],
        "Training innovations",
    )

    plot_innovations(
        df_test,
        test_ekf["innovations"],
        "Testing innovations",
    )

    plot_NIS(
        df_train,
        train_ekf["nis"],
        "Training NIS",
    )

    plot_NIS(
        df_test,
        test_ekf["nis"],
        "Testing NIS",
    )

    plot_full_ekf_comparison(
        df_train,
        x_sim_train,
        train_ekf["x_pred"],
        train_ekf["x_hat"],
        title="Training: Model vs EKF",
    )

    plot_full_ekf_comparison(
        df_test,
        x_sim_test,
        test_ekf["x_pred"],
        test_ekf["x_hat"],
        title="Testing: Model vs EKF",
    )

    plot_residuals_vs_time(
        df_train,
        x_sim_train,
        df_test,
        x_sim_test,
        window="6h",
        title="Open-loop residual"
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
