import pandas as pd

from Methods.Blackbox.sequence_model_training import train_sequence_model
from Methods.Blackbox.RNN.rnn_model import RNNBaseline
from Methods.Blackbox.experiment_configs import (
    DEVICE,
    SENSOR_COLS,
    RENAME_MAP,
    INPUT_COLS,
    TARGET_COLS,
    RNN_MODEL_CONFIG,
    RNN_MODEL_TYPE,
    TRAINING_CONFIG,
    TS_COL,
    WINDOW_STEPS,
    SEED,
)
from scripts.misc.feature_engineering import ts_cols
from Methods.Blackbox.profile_dataset import (
    create_profiles,
    scale_train_val_test_data,
    tensorize_profiles,
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
    """Load one split, select the required sensors, and rename them."""
    data = pd.read_csv(csv_path)
    data, dt_s = ts_cols(data, TS_COL)

    required_cols = [TS_COL, *SENSOR_COLS]
    missing_cols = [
        column
        for column in required_cols
        if column not in data.columns
    ]

    if missing_cols:
        raise KeyError(
            f"Missing required columns in {csv_path}: {missing_cols}"
        )

    data = data[required_cols].copy()
    data = data.rename(columns=RENAME_MAP)

    return data, dt_s


def main() -> None:
    set_reproducibility(SEED)

    # -----------------------------------------------------
    # Load the three pre-split datasets
    # -----------------------------------------------------
    train_path = "../ds1_generator_train.csv"
    val_path = "../ds1_generator_val.csv"
    test_path = "../ds1_generator_test.csv"


    train_data, train_dt_s = load_and_prepare_data(train_path)
    val_data, val_dt_s = load_and_prepare_data(val_path)
    test_data, test_dt_s = load_and_prepare_data(test_path)

    dt_s = train_dt_s  # sampling interval (same across splits)

    # -----------------------------------------------------
    # Profile each split independently
    # -----------------------------------------------------
    train_data, train_profiles = create_profiles(
        train_data,
        ts_col=TS_COL,
        window_steps=WINDOW_STEPS,
        dt_s=train_dt_s,
    )

    val_data, val_profiles = create_profiles(
        val_data,
        ts_col=TS_COL,
        window_steps=WINDOW_STEPS,
        dt_s=val_dt_s,
    )

    test_data, test_profiles = create_profiles(
        test_data,
        ts_col=TS_COL,
        window_steps=WINDOW_STEPS,
        dt_s=test_dt_s,
    )

    print(f"Training profiles:   {len(train_profiles)}")
    print(f"Validation profiles: {len(val_profiles)}")
    print(f"Test profiles:       {len(test_profiles)}")

    # -----------------------------------------------------
    # Fit scalers on training data and apply to all splits
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
        train_data,
        train_profiles,
        INPUT_COLS,
        TARGET_COLS,
        device=DEVICE,
    )

    x_val, y_val, mask_val = tensorize_profiles(
        val_data,
        val_profiles,
        INPUT_COLS,
        TARGET_COLS,
        device=DEVICE,
    )

    x_test, y_test, mask_test = tensorize_profiles(
        test_data,
        test_profiles,
        INPUT_COLS,
        TARGET_COLS,
        device=DEVICE,
    )

    print(f"x_train shape: {tuple(x_train.shape)}")
    print(f"x_val shape:   {tuple(x_val.shape)}")
    print(f"x_test shape:  {tuple(x_test.shape)}")

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = RNNBaseline(
        **RNN_MODEL_CONFIG
    ).to(DEVICE)

    # -----------------------------------------------------
    # Training and best-validation checkpoint selection
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
    # Final test evaluation
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
        data=test_data,
        test_profiles=test_profiles,
        target_cols=TARGET_COLS,
        y_scaler=y_scaler,
        ts_col=TS_COL,
    )

    # -----------------------------------------------------
    # Save model and preprocessing metadata
    # -----------------------------------------------------
    model_name = RNN_MODEL_TYPE.lower()

    save_sequence_model(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        input_cols=INPUT_COLS,
        target_cols=TARGET_COLS,
        model_type=RNN_MODEL_TYPE,
        model_config=RNN_MODEL_CONFIG,
        training_config={
            **TRAINING_CONFIG,
            "seed": SEED,
        },
        dt_s=dt_s,
        path=f"{model_name}_winding_baseline.pt",
        history=history,
    )

if __name__ == "__main__":
    main()