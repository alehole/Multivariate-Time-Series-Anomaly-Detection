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


CSV_FULL = "toy_generator_1state.csv"
CSV_TRAIN = "toy_generator_1state_train.csv"
CSV_TEST = "toy_generator_1state_test.csv"

# ---------------------------------------------------------
# SDE run configuration
# ---------------------------------------------------------
RUN_TOY_CHECK = True
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
    CSV_TRAIN = cfg.DATA_PATH /"toy_sim/toy_generator_train.csv"
    CSV_TEST = cfg.DATA_PATH / "toy_sim/toy_generator_test.csv"
else:
    CSV_TRAIN = cfg.DATA_PATH / "train_test_split" / "ds1_generator_train.csv"
    CSV_TEST = cfg.DATA_PATH / "train_test_split"/ "ds1_generator_test.csv"

MODEL_CONFIGS = {
    "1state": {
        "state_cols": ["T1"],
        "meas_cols": ["T1"],
        "input_cols": ["P", "Tref"],
        "parameter_names": ["C1", "R1"],
        "theta0": np.array([115_000.0, 0.04]),
        "lower_bound": np.array([1.0e4, 0.005]),
        "upper_bound": np.array([1.0e7, 0.5]),
        "Q": np.diag([0.0001]),
        "R": np.diag([0.5**2]),
        "C": np.array([[1.0]]),
    },
    "2state": {
        "state_cols": ["T1", "T2"],
        "meas_cols": ["T1"],
        "input_cols": ["P", "Tref"],
        "parameter_names": ["C1", "C2", "R1", "R2"],
        "theta0": np.array([
            1.0e5,
            5.0e5,
            0.04,
            0.10,
        ]),
        "lower_bound": np.array([
            1.0e4,
            1.0e4,
            0.005,
            0.005,
        ]),
        "upper_bound": np.array([
            1.0e7,
            1.0e8,
            0.5,
            1.0,
        ]),
        "Q": np.diag([0.0001, 0.0001]),
        "R": np.diag([0.5**2]),
        "C": np.array([[1.0, 0.0]]),
    },
}
selected = MODEL_CONFIGS[MODEL_OPTION]

STATE_COLS = selected["state_cols"]
MEAS_COLS = selected["meas_cols"]
INPUT_COLS = selected["input_cols"]
PARAMETER_NAMES = selected["parameter_names"]
THETA0 = selected["theta0"]
LOWER_BOUND = selected["lower_bound"]
UPPER_BOUND = selected["upper_bound"]
Q = selected["Q"]
R = selected["R"]
C = selected["C"]

# ---------------------------------------------------------
# Dataset sensor mapping
# ---------------------------------------------------------
if RUN_TOY_CHECK:
    RENAME_MAP = {
        "T1": "T1",
        "P": "P",
        "Tref": "Tref",
    }

else:
    RENAME_MAP = {
        "AE PORT GEN.U-WINDING TEMP.": "T1",
        "POWER_kW": "P",
        "AE PORT HT FW OUTLET TEMP.": "Tref",
    }

SENSOR_COLS = list(RENAME_MAP.keys())
