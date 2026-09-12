import faulthandler
import pandas as pd
from pathlib import Path

from Methods.Blackbox.sequence_model_training import train_sequence_model
from Methods.Blackbox.TCN.TCN import TCNBaseline
from Methods.Blackbox.experiment_configs import (
    TCN_MODEL_CONFIG,
    TRAINING_CONFIG,
    WINDOW_STEPS,
    TCN_MODEL_TYPE,
)
from scripts.misc.feature_engineering import ts_cols
from Methods.Blackbox.profile_dataset import (
    tensorize_profiles,
    create_profiles,
    scale_train_val_test_data,
)
from Methods.Blackbox.common_BB_scripts import (
    save_sequence_model,
    evaluate_sequence_model,
    plot_sequence_model_results,
    print_metrics,
    set_reproducibility,
)
import config as cfg

def load_and_prepare_data(
    csv_path: Path,
) -> tuple[pd.DataFrame, float]:
    """Load, select, and rename the required variables."""
    data = pd.read_csv(csv_path)
    data, dt_s = ts_cols(data, cfg.TS_COL)


    data = data[[cfg.TS_COL, *cfg.SENSOR_COLS]].copy()
    data = data.rename(columns=cfg.RENAME_MAP)

    return data, dt_s
def main():
    set_reproducibility(cfg.SEED)
    model_config = cfg.CONFIG
    data_path = Path(cfg.DATA_PATH)

    # -----------------------------------------------------
    # Load the three pre-split datasets
    # -----------------------------------------------------

    train_path = data_path / "train_test_split" / f"{cfg.DS}_generator_train.csv"
    val_path = data_path / "train_test_split" / f"{cfg.DS}_generator_val.csv"
    test_path = data_path / "train_test_split" / f"{cfg.DS}_generator_test.csv"

    train_data, train_dt_s = load_and_prepare_data(train_path)
    val_data,   val_dt_s   = load_and_prepare_data(val_path)
    test_data,  test_dt_s  = load_and_prepare_data(test_path)

    dt_s = train_dt_s  # sampling interval (same across splits)

    # -----------------------------------------------------
    # Profile each split independently
    # -----------------------------------------------------
    train_data, train_profiles = create_profiles(
        train_data, ts_col=cfg.TS_COL, window_steps=WINDOW_STEPS, dt_s=train_dt_s,
    )
    val_data, val_profiles = create_profiles(
        val_data, ts_col=cfg.TS_COL, window_steps=WINDOW_STEPS, dt_s=val_dt_s,
    )
    test_data, test_profiles = create_profiles(
        test_data, ts_col=cfg.TS_COL, window_steps=WINDOW_STEPS, dt_s=test_dt_s,
    )
    # -----------------------------------------------------
    # Scaling (fit on train only, apply to all three)
    # -----------------------------------------------------
    (
        train_data,
        val_data,
        test_data,
        x_scaler,
        y_scaler,
    ) = scale_train_val_test_data(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        input_cols=cfg.INPUT_COLS,
        target_cols=cfg.TARGET_COLS,
    )

    # -----------------------------------------------------
    # Tensor creation
    # -----------------------------------------------------
    x_train, y_train, mask_train = tensorize_profiles(
        train_data, train_profiles, cfg.INPUT_COLS, cfg.TARGET_COLS, device=cfg.DEVICE
    )
    x_val, y_val, mask_val = tensorize_profiles(
        val_data, val_profiles, cfg.INPUT_COLS, cfg.TARGET_COLS, device=cfg.DEVICE
    )
    x_test, y_test, mask_test = tensorize_profiles(
        test_data, test_profiles, cfg.INPUT_COLS, cfg.TARGET_COLS, device=cfg.DEVICE
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = TCNBaseline(**TCN_MODEL_CONFIG).to(cfg.DEVICE)

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    model, history = train_sequence_model(
        model=model,
        x_train=x_train, y_train=y_train, mask_train=mask_train,
        x_val=x_val, y_val=y_val, mask_val=mask_val,
        **TRAINING_CONFIG,
    )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------
    metrics = evaluate_sequence_model(
        model=model,
        x_test=x_test, y_test=y_test, mask_test=mask_test,
        y_scaler=y_scaler,
    )
    print_metrics(metrics)
    # -----------------------------------------------------
    # Plotting
    # -----------------------------------------------------
    plot_sequence_model_results(
        model=model,
        x_test=x_test,
        mask_test=mask_test,
        data=test_data,
        test_profiles=test_profiles,
        target_cols=cfg.TARGET_COLS,
        y_scaler=y_scaler,
        ts_col=cfg.TS_COL,
        show_threshold=False,
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------
    save_sequence_model(
        model=model,
        x_scaler=x_scaler, y_scaler=y_scaler,
        input_cols=cfg.INPUT_COLS, target_cols=cfg.TARGET_COLS,
        model_type=TCN_MODEL_TYPE, model_config=TCN_MODEL_CONFIG,
        training_config={**TRAINING_CONFIG, "seed": cfg.SEED},
        dt_s=dt_s,
        path=f"{cfg.DS}_{model_config}_tcn_winding_baseline.pt",
        history=history,
    )

if __name__ == "__main__":
    main()