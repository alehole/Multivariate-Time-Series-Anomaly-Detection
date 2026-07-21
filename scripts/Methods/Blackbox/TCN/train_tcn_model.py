import pandas as pd

from Methods.Blackbox.sequence_model_training import train_sequence_model
from Methods.Blackbox.TCN.TCN import TCNBaseline
from Methods.Blackbox.experiment_configs import (
    DEVICE,
    SENSOR_COLS,
    RENAME_MAP,
    INPUT_COLS,
    TARGET_COLS,
    MODEL_CONFIG,
    TRAINING_CONFIG,
    TS_COL,
    WINDOW_STEPS,
    VAL_PROFILE_LEN,
    TEST_PROFILE_LEN,
    MODEL_TYPE,
    SEED,
)
from scripts.misc.feature_engineering import ts_cols
from Methods.Blackbox.profile_dataset import (
    train_val_test_split_profiles,
    scale_profile_data,
    tensorize_profiles
)
from Methods.Blackbox.common_BB_scripts import (
    save_sequence_model,
    evaluate_sequence_model,
    plot_sequence_model_results,
    print_metrics,
    set_reproducibility,
)

def main():
    set_reproducibility(SEED)

    csv_path = "AE_PORT.csv"
    data = pd.read_csv(csv_path)
    data, dt_s = ts_cols(data, TS_COL)
    data = data[[TS_COL, *SENSOR_COLS]].copy()
    data = data.rename(columns=RENAME_MAP)
    # -----------------------------------------------------
    # Train-validation-test split
    # -----------------------------------------------------
    data, train_profiles, val_profiles, test_profiles, profile_sizes = train_val_test_split_profiles(
        data,
        ts_col=TS_COL,
        window_steps=WINDOW_STEPS,
        val_profile_len=VAL_PROFILE_LEN,
        test_profile_len=TEST_PROFILE_LEN,
        dt_s=dt_s,
    )
    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------
    data, x_scaler, y_scaler = scale_profile_data(
        data,
        train_profiles=train_profiles,
        input_cols=INPUT_COLS,
        target_cols=TARGET_COLS,
    )
    # -----------------------------------------------------
    # Tensor creation
    # -----------------------------------------------------
    x_train, y_train, mask_train = tensorize_profiles(
        data, train_profiles, INPUT_COLS, TARGET_COLS, device=DEVICE
    )

    x_val, y_val, mask_val = tensorize_profiles(
        data, val_profiles, INPUT_COLS, TARGET_COLS, device=DEVICE
    )

    x_test, y_test, mask_test = tensorize_profiles(
        data, test_profiles, INPUT_COLS, TARGET_COLS, device=DEVICE
    )
    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = TCNBaseline(
        **MODEL_CONFIG
    ).to(DEVICE)

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    model, history = train_sequence_model(
        model=model,
        x_train=x_train,
        y_train=y_train,
        mask_train=mask_train,
        x_val=x_val,
        y_val=y_val,
        mask_val=mask_val,
        **TRAINING_CONFIG,
    )
    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------
    metrics = evaluate_sequence_model(
        model=model,
        x_test=x_test,
        y_test=y_test,
        mask_test=mask_test,
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
        data=data,
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
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        input_cols=INPUT_COLS,
        target_cols=TARGET_COLS,
        model_type=MODEL_TYPE,
        model_config=MODEL_CONFIG,
        training_config={
            **TRAINING_CONFIG,
            "seed": SEED,
        },
        dt_s=dt_s,
        path="tcn_winding_baseline.pt",
        history=history,
    )

if __name__ == "__main__":
    main()