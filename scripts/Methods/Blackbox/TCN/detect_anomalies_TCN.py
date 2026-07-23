from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn

import config as cfg
from Methods.Blackbox.TCN.TCN import TCNBaseline
from Methods.Blackbox.experiment_configs import (
    RENAME_MAP,
    SENSOR_COLS,
    TS_COL,
    WINDOW_STEPS,
)
from Methods.Blackbox.profile_dataset import (
    create_profiles,
    tensorize_profiles,
)
from Methods.Blackbox.common_BB_scripts import predict_sequence_model, load_sequence_model
from scripts.misc.feature_engineering import ts_cols


def load_trained_tcn(
    model_path: str | Path,
    device: torch.device,
):
    model_registry = {
        "TCN": TCNBaseline,
        "TCNBaseline": TCNBaseline,
    }

    return load_sequence_model(
        path=model_path,
        model_registry=model_registry,
        device=device,
    )

def main():
    ANOMALY_THRESHOLD = 5
    PREDICTION_HOURS_START = 0  # start prediction X hours into dataset
    PREDICTION_HOURS = 120  # predict for next X hours
    # -----------------------------------------------------
    # 1 Read pre trained model
    # -----------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, metadata = load_trained_tcn(
        model_path="tcn_winding_baseline.pt",
        device=device,
    )

    x_scaler = metadata["x_scaler"]
    y_scaler = metadata["y_scaler"]
    input_cols = metadata["input_cols"]
    target_cols = metadata["target_cols"]
    dt_s = metadata["dt_s"]

    print("Model loaded successfully")
    print("Input columns:", input_cols)
    print("Target columns:", target_cols)
    print("Sampling interval:", dt_s)

    # -----------------------------------------------------
    # 2 Load dataset
    # -----------------------------------------------------
    data_path = Path(cfg.DATA_PATH)
    train_path = data_path / "train_test_split/with_anomalies/ds1_generator_test_w_anomalies.csv"


if __name__ == "__main__":
    main()
