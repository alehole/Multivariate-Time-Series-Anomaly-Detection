from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch

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
from Methods.Blackbox.common_BB_scripts import predict_sequence_model
from scripts.misc.feature_engineering import ts_cols



def main():
    ANOMALY_THRESHOLD = 5
    PREDICTION_HOURS_START = 0  # start prediction X hours into dataset
    PREDICTION_HOURS = 120  # predict for next X hours




if __name__ == "__main__":
    main()
