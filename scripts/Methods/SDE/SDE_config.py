import numpy as np
import config as cfg
from pathlib import Path

# ---------------------------------------------------------
# SDE model selection
# ---------------------------------------------------------
MODEL_OPTION = "1state"
# ---------------------------------------------------------
# Toy-data configuration
# ---------------------------------------------------------
RUN_TOY_CHECK = True

CSV_FULL = "toy_generator_1state.csv"
CSV_TRAIN = "toy_generator_1state_train.csv"
CSV_TEST = "toy_generator_1state_test.csv"

# ---------------------------------------------------------
# SDE run configuration
# ---------------------------------------------------------
RUN_PL1 = False
RUN_PL2 = False
RUN_MCMC = False
RUN_WILKS = False

MAXITER = 2000
NIS_THRESHOLD_PERCENTILE = 0.995

data_path = Path(cfg.DATA_PATH)
train_path = data_path / "train_test_split/ds1_generator_train.csv"
val_path = data_path / "train_test_split/ds1_generator_val.csv"
test_path = data_path / "train_test_split/ds1_generator_test.csv"

if RUN_TOY_CHECK:
    CSV_TRAIN = cfg.DATA_PATH / "toy_generator_train.csv"
    CSV_TEST = cfg.DATA_PATH / "toy_generator_test.csv"
else:
    CSV_TRAIN = (
        cfg.DATA_PATH
        / "train_test_split"
        / "ds1_generator_train.csv"
    )
    CSV_TEST = (
        cfg.DATA_PATH
        / "train_test_split"
        / "ds1_generator_test.csv"
    )

# ---------------------------------------------------------
# One-state thermal model
# ---------------------------------------------------------
STATE_COLS = ["T1"]
MEAS_COLS = ["T1"]
INPUT_COLS = ["P", "Tref"]

SENSOR_COLS = [
    "T1",
    "P",
    "Tref",
]

RENAME_MAP = {
    "T1": "T1",
    "P": "P",
    "Tref": "Tref",
}

PARAMETER_NAMES = [
    "C1",
    "R1",
]
# ---------------------------------------------------------
# Parameter-estimation configuration
# ---------------------------------------------------------
THETA0 = np.array([
    1.0e5,  # C1 [kJ/°C]
    0.05,   # R1 [°C/kW]
])

LOWER_BOUND = np.array([
    1.0e4,
    0.005,
])

UPPER_BOUND = np.array([
    1.0e7,
    0.5,
])

# ---------------------------------------------------------
# EKF configuration
# ---------------------------------------------------------
Q = np.diag([
    0.0001,
])

R = np.diag([
    0.25**2,
])

C = np.array([
    [1.0],
])