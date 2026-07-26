from misc.data import load_data
from pathlib import Path
import config as cfg
import SDE_config as sde_cfg
from notify_phone import notify_phone
from parameter_estimation import estimate_parameters_mle

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

    # ------------------------------------------------------------
    # Estimate grey-box model parameters from training data
    # ------------------------------------------------------------
    theta_hat, result = estimate_parameters_mle(df_train)
    print("\nEstimated parameters:")
    for name, value in zip(sde_cfg.PARAMETER_NAMES, theta_hat):
        print(f"{name}: {value:.6g}")

    print("Neg log-likelihood:", result.fun)
    print("Iterations:", result.get("nfev", "n/a"))
    print(result.message)


if __name__ == "__main__":
    main()
    if sde_cfg.RUN_PL1 or sde_cfg.RUN_PL2 or sde_cfg.RUN_MCMC:
        notify_phone(
            "Pipeline has finished.",
            title="EKF finished",
        )
