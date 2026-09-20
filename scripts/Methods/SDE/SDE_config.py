import numpy as np
import config as cfg
from pathlib import Path

# ---------------------------------------------------------
# SDE model selection
# ---------------------------------------------------------
MODEL_OPTION = "1state"

# ---------------------------------------------------------
# SDE run configuration
# ---------------------------------------------------------
RUN_TOY_CHECK = False
RUN_PL1 = False
RUN_PL2 = False
RUN_WILKS = False

RESIDUAL_ANOMALY_THRESHOLD = 1.95  # [°C]
MAXITER = 2000
NIS_THRESHOLD_PERCENTILE = 0.995


# ---------------------------------------------------------
# Toy-data configuration
# ---------------------------------------------------------
TOY_THETA_TRUE = np.array([
    120_000.0,  # C1 [kJ/°C]
    0.05,       # R1 [°C/kW]
])

# ---------------------------------------------------------
# SDE thermal-model configurations
# ---------------------------------------------------------

MODEL_CONFIGS = {
    "1state": {
        "state_cols": ["T1"],
        "meas_cols": ["T1"],
        "input_cols": ["P", "Tref"],
        "parameter_names": ["C1", "R1"],
        "theta0": np.array([2e5, 0.04]),
       # "lower_bound": np.array([1.0e4, 0.005]),
       # "upper_bound": np.array([1.0e6, 0.5]),
        "lower_bound": np.array([1.0e4, 0.005]),
        "upper_bound": np.array([1.0e7, 1.0]),
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
# Dataset paths
# ---------------------------------------------------------

if RUN_TOY_CHECK:
    CSV_TRAIN = cfg.DATA_PATH / "toy_sim" / "toy_generator_train.csv"
    CSV_TEST = cfg.DATA_PATH / "toy_sim" / "toy_generator_test.csv"
else:
    CSV_TRAIN = cfg.TRAIN_PATH
    CSV_TEST = cfg.TEST_PATH

# ---------------------------------------------------------
# SDE-specific variable mapping
# ---------------------------------------------------------
if RUN_TOY_CHECK:
    RENAME_MAP = {
        "T1": "T1",
        "P": "P",
        "Tref": "Tref",
    }

else:
    generic_to_raw = {
        generic: raw
        for raw, generic in cfg.RENAME_MAP.items()
    }

    RENAME_MAP = {
        generic_to_raw["T1"]: "T1",
        generic_to_raw["E1"]: "P",
        generic_to_raw["T17"]: "Tref",
    }

SENSOR_COLS = list(RENAME_MAP.keys())
