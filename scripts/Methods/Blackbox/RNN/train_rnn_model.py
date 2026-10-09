import pandas as pd
from pathlib import Path
import itertools
import random
import torch
import sys

from Methods.Blackbox.sequence_model_training import train_sequence_model
from Methods.Blackbox.RNN.RNN import RNNBaseline
from Methods.Blackbox.experiment_configs import (
    RNN_MODEL_CONFIG,
    RNN_MODEL_TYPE,
    TRAINING_CONFIG,
    WINDOW_STEPS,
    RNN_SEARCH_SPACE,
    TRAINING_SEARCH_SPACE,
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
import config as cfg

def load_and_prepare_data(
    csv_path: Path,
) -> tuple[pd.DataFrame, float]:
    """Load one split, select the required sensors, and rename them."""
    data = pd.read_csv(csv_path)
    data, dt_s = ts_cols(data, cfg.TS_COL)

    required_cols = [cfg.TS_COL, *cfg.SENSOR_COLS]
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
    data = data.rename(columns=cfg.RENAME_MAP)

    return data, dt_s

def random_search_rnn(
    x_train,
    y_train,
    mask_train,
    x_val,
    y_val,
    mask_val,
    n_trials=20,
    seed=42,
):
    """
    Randomly evaluate a subset of the discrete hyperparameter grid.
    Hyperparameters are selected using validation loss only.
    """

    # Combine architecture and training hyperparameters
    search_space = {
        **RNN_SEARCH_SPACE,
        **TRAINING_SEARCH_SPACE,
    }

    keys = list(search_space.keys())

    combinations = list(
        itertools.product(
            *[search_space[key] for key in keys]
        )
    )

    rng = random.Random(seed)
    rng.shuffle(combinations)

    combinations = combinations[
        :min(n_trials, len(combinations))
    ]

    results = []

    # ---------------------------------------------------------
    # Hyperparameter trials
    # ---------------------------------------------------------
    for trial, values in enumerate(combinations, start=1):

        params = dict(zip(keys, values))

        print(
            f"\n{'=' * 60}\n"
            f"Trial {trial}/{len(combinations)}\n"
            f"{params}\n"
            f"{'=' * 60}"
        )

        # Use a deterministic but different seed per trial
        set_reproducibility(seed)

        model_config = {
            **RNN_MODEL_CONFIG,
            "hidden_size": params["hidden_size"],
            "num_layers": params["num_layers"],
            "dropout": params["dropout"],
        }

        training_config = {
            **TRAINING_CONFIG,
            "lr": params["lr"],
            "weight_decay": params["weight_decay"],
        }

        model = RNNBaseline(
            **model_config
        ).to(cfg.DEVICE)

        model, history = train_sequence_model(
            model=model,
            x_train=x_train,
            y_train=y_train,
            mask_train=mask_train,
            x_val=x_val,
            y_val=y_val,
            mask_val=mask_val,
            **training_config,
        )

        # Best validation loss obtained during training
        best_val_loss = float(min(history["val_loss"]))

        results.append({
            **params,
            "best_val_loss": best_val_loss,
        })

        print(
            f"Best validation loss: "
            f"{best_val_loss:.6f}"
        )

        # Free GPU memory before next trial
        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Rank configurations
    # ---------------------------------------------------------
    results = pd.DataFrame(results)

    results = results.sort_values(
        "best_val_loss"
    ).reset_index(drop=True)

    print("\nHyperparameter search results:")
    print(results)

    best = results.iloc[0].to_dict()

    print("\nBest configuration:")
    print(best)

    return results, best

def main():
    set_reproducibility(cfg.SEED)
    model_config = cfg.CONFIG

    data_path = Path(cfg.DATA_PATH)

    # -----------------------------------------------------
    # Load the three pre-split datasets
    # -----------------------------------------------------
    train_path = data_path/"train_test_split"/f"{cfg.DS}_generator_train.csv"
    val_path = data_path/"train_test_split"/ f"{cfg.DS}_generator_val.csv"
    test_path = data_path/"train_test_split"/ f"{cfg.DS}_generator_test.csv"

    train_data, train_dt_s = load_and_prepare_data(train_path)
    val_data, val_dt_s = load_and_prepare_data(val_path)
    test_data, test_dt_s = load_and_prepare_data(test_path)

    dt_s = train_dt_s  # sampling interval (same across splits)



    # -----------------------------------------------------
    # Profile each split independently
    # -----------------------------------------------------
    train_data, train_profiles = create_profiles(
        train_data,
        ts_col=cfg.TS_COL,
        window_steps=WINDOW_STEPS,
        dt_s=train_dt_s,
    )

    val_data, val_profiles = create_profiles(
        val_data,
        ts_col=cfg.TS_COL,
        window_steps=WINDOW_STEPS,
        dt_s=val_dt_s,
    )

    test_data, test_profiles = create_profiles(
        test_data,
        ts_col=cfg.TS_COL,
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
        input_cols=cfg.INPUT_COLS,
        target_cols=cfg.TARGET_COLS,
    )

    # -----------------------------------------------------
    # Tensor creation
    # -----------------------------------------------------
    x_train, y_train, mask_train = tensorize_profiles(
        train_data,
        train_profiles,
        cfg.INPUT_COLS,
        cfg.TARGET_COLS,
        device=cfg.DEVICE,
    )

    x_val, y_val, mask_val = tensorize_profiles(
        val_data,
        val_profiles,
        cfg.INPUT_COLS,
        cfg.TARGET_COLS,
        device=cfg.DEVICE,
    )

    x_test, y_test, mask_test = tensorize_profiles(
        test_data,
        test_profiles,
        cfg.INPUT_COLS,
        cfg.TARGET_COLS,
        device=cfg.DEVICE,
    )

    print(f"x_train shape: {tuple(x_train.shape)}")
    print(f"x_val shape:   {tuple(x_val.shape)}")
    print(f"x_test shape:  {tuple(x_test.shape)}")
    # -----------------------------------------------------
    # Random grid search
    # -----------------------------------------------------
    GRID_SEARCH=True
    if GRID_SEARCH:
        search_results, best = random_search_rnn(
            x_train=x_train,
            y_train=y_train,
            mask_train=mask_train,
            x_val=x_val,
            y_val=y_val,
            mask_val=mask_val,
            n_trials=100,
            seed=cfg.SEED,
        )
        search_results.to_csv(
            "rnn_hyperparameter_search.csv",
            index=False,
        )
        sys.exit(0)

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = RNNBaseline(
        **RNN_MODEL_CONFIG
    ).to(cfg.DEVICE)

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
        target_cols=cfg.TARGET_COLS,
        y_scaler=y_scaler,
        ts_col=cfg.TS_COL,
        show_threshold=False,
    )

    # -----------------------------------------------------
    # Save model and preprocessing metadata
    # -----------------------------------------------------
    model_name = RNN_MODEL_TYPE.lower()


    save_sequence_model(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        input_cols=cfg.INPUT_COLS,
        target_cols=cfg.TARGET_COLS,
        model_type=RNN_MODEL_TYPE,
        model_config=RNN_MODEL_CONFIG,
        training_config={
            **TRAINING_CONFIG,
            "seed": cfg.SEED,
        },
        dt_s=dt_s,
        path=f"{cfg.DS}_{model_config}_{model_name}_winding_baseline.pt",
        history=history,
    )

if __name__ == "__main__":
    main()