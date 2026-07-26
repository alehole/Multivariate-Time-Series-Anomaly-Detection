from misc.data import load_data
from pathlib import Path
import config as cfg
import SDE_config as sde_cfg
from notify_phone import notify_phone
from parameter_estimation import estimate_parameters_mle
from PL1 import *
from PL2 import *
from plotting import *
from model import simulate_model


def main():
    data_path = Path(cfg.DATA_PATH)
    # -----------------------------------------------------
    # Load training and testing datasets
    # -----------------------------------------------------
    if sde_cfg.RUN_TOY_CHECK:
        train_path = data_path / "toy_sim/toy_generator_train.csv"
        test_path  = data_path / "toy_sim/toy_generator_test.csv"
    else:
        train_path = data_path / "train_test_split/ds1_generator_train.csv"
        val_path   = data_path / "train_test_split/ds1_generator_val.csv"
        test_path  = data_path / "train_test_split/ds1_generator_test.csv"

    df_train = load_data(train_path, cfg.TS_COL, sde_cfg.SENSOR_COLS, sde_cfg.RENAME_MAP)
    df_test = load_data(test_path, cfg.TS_COL, sde_cfg.SENSOR_COLS, sde_cfg.RENAME_MAP)
    print(f"Training observations: {len(df_train):,}")
    print(f"Test observations:     {len(df_test):,}")
    print(f"Selected model:        {sde_cfg.MODEL_OPTION}")

    # -----------------------------------------------------
    # Estimate grey-box model parameters from training
    # dataset by minimizing the EKF likelihood cost.
    # -----------------------------------------------------
    theta_hat, result = estimate_parameters_mle(df_train)
    # -----------------------------------------------------
    # Report estimated physical parameters
    # -----------------------------------------------------
    print("\nEstimated parameters:")
    for name, value in zip(sde_cfg.PARAMETER_NAMES, theta_hat):
        print(f"{name}: {value:.6g}")

    print("\nOptimization diagnostics:")
    print("Success:", result.success)
    print("Message:", result.message)
    print("Final likelihood cost:", result.fun)
    print("Iterations:", result.get("nit", "n/a"))
    print("Function evaluations:", result.get("nfev", "n/a"),
    )
    if sde_cfg.RUN_TOY_CHECK:
        # For the toy, compare against known truth
        from toy_simulation import theta_true
        print("\nRecovery check (est / true):")
        for name, est, true in zip(sde_cfg.PARAMETER_NAMES, theta_hat, theta_true):
            print(f"  {name}: {est:.5g} vs {true:.5g}  ({100 * est / true:.3f}%)")

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
    if sde_cfg.RUN_PL1:
        profiles = run_pl1(df_train, theta_hat, nll_ref)



    # ------------------------------------------------------------
    # Open-loop simulation (no measurement correction)
    # ------------------------------------------------------------
    x_sim_train = simulate_model(df_train, theta_hat)
    x_sim_test = simulate_model(df_test, theta_hat)

    plot_simulated_vs_actual(df_train, x_sim_train, "Training")
    plot_simulated_vs_actual(df_test, x_sim_test, "Testing")






if __name__ == "__main__":
    main()
    if sde_cfg.RUN_PL1 or sde_cfg.RUN_PL2 or sde_cfg.RUN_MCMC:
        notify_phone(
            "Pipeline has finished.",
            title="EKF finished",
        )
