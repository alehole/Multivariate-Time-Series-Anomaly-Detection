import pandas as pd

from Methods.Blackbox.sequence_model_training import train_sequence_model
from Methods.Blackbox.TCN.TCN import TCNBaseline
from Methods.Blackbox.experiment_configs import (
    DEVICE,
    SENSOR_COLS,
    RENAME_MAP,
    INPUT_COLS,
    TARGET_COLS,
    TCN_MODEL_CONFIG,
    TRAINING_CONFIG,
    TS_COL,
    WINDOW_STEPS,
    TCN_MODEL_TYPE,
    SEED,
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

def load_and_prepare_data(
    csv_path: str,
) -> tuple[pd.DataFrame, float]:
    """Load, select, and rename the required variables."""
    data = pd.read_csv(csv_path)
    data, dt_s = ts_cols(data, TS_COL)


    data = data[[TS_COL, *SENSOR_COLS]].copy()
    data = data.rename(columns=RENAME_MAP)

    return data, dt_s

def main():
    set_reproducibility(SEED)

    # -----------------------------------------------------
    # Load the three pre-split datasets
    # -----------------------------------------------------
    train_path = "ds1_generator_train.csv"
    val_path   = "ds1_generator_val.csv"
    test_path  = "ds1_generator_test.csv"

    train_data, train_dt_s = load_and_prepare_data(train_path)
    val_data,   val_dt_s   = load_and_prepare_data(val_path)
    test_data,  test_dt_s  = load_and_prepare_data(test_path)

    dt_s = train_dt_s  # sampling interval (same across splits)

    # -----------------------------------------------------
    # Profile each split independently
    # -----------------------------------------------------
    train_data, train_profiles = create_profiles(
        train_data, ts_col=TS_COL, window_steps=WINDOW_STEPS, dt_s=train_dt_s,
    )
    val_data, val_profiles = create_profiles(
        val_data, ts_col=TS_COL, window_steps=WINDOW_STEPS, dt_s=val_dt_s,
    )
    test_data, test_profiles = create_profiles(
        test_data, ts_col=TS_COL, window_steps=WINDOW_STEPS, dt_s=test_dt_s,
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
        input_cols=INPUT_COLS,
        target_cols=TARGET_COLS,
    )

    # -----------------------------------------------------
    # Tensor creation
    # -----------------------------------------------------
    x_train, y_train, mask_train = tensorize_profiles(
        train_data, train_profiles, INPUT_COLS, TARGET_COLS, device=DEVICE
    )
    x_val, y_val, mask_val = tensorize_profiles(
        val_data, val_profiles, INPUT_COLS, TARGET_COLS, device=DEVICE
    )
    x_test, y_test, mask_test = tensorize_profiles(
        test_data, test_profiles, INPUT_COLS, TARGET_COLS, device=DEVICE
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = TCNBaseline(**TCN_MODEL_CONFIG).to(DEVICE)

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
        target_cols=TARGET_COLS,
        y_scaler=y_scaler,
        ts_col=TS_COL,
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------
    save_sequence_model(
        model=model,
        x_scaler=x_scaler, y_scaler=y_scaler,
        input_cols=INPUT_COLS, target_cols=TARGET_COLS,
        model_type=TCN_MODEL_TYPE, model_config=TCN_MODEL_CONFIG,
        training_config={**TRAINING_CONFIG, "seed": SEED},
        dt_s=dt_s,
        path="tcn_winding_baseline.pt",
        history=history,
    )

if __name__ == "__main__":
    main()