import numpy as np
import config as cfg

# ---------------------------------------------------------
# SDE model selection
# ---------------------------------------------------------
MODEL_OPTION = "2state_B"

# ---------------------------------------------------------
# SDE run configuration
# ---------------------------------------------------------
RUN_TOY_CHECK = False
RUN_PL1 = True
RUN_PL2 = True
RUN_WILKS = False
ESTIMATE_Q = False

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
        "theta0": np.array([1.15e5, 0.04]),
        "lower_bound": np.array([1.0e4, 0.005]),
        "upper_bound": np.array([1.0e6, 0.5]),
        #"Q": np.diag([0.0001]),
        "Q_INIT": np.diag([0.009]),
        #"R": np.diag([0.5**2]),
        "R": np.array([[0.0225]]),  # sigma = 0.15 °C
        "C": np.array([[1.0]]),
    },

    "2state_A": {
        "state_cols": ["T1", "T2"],
        "meas_cols": ["T1", "T2"],
        "input_cols": ["P", "Tref"],
        "parameter_names": ["C1", "C2", "R1", "R2"],
        "theta0": np.array([
            1.0e6,  # C1
            1.0e6,  # C2
            0.02,  # R1
            0.02,  # R2
        ]),
        "lower_bound": np.array([
            1.0e4,
            1.0e4,
            0.005,
            0.005,
        ]),
        "upper_bound": np.array([
            1e8,  # C1
            5e9,  # C2
            0.5,  # R1
            2.0,  # R2
        ]),
        "Q_INIT": np.diag([
            0.0001,
            0.0001,
        ]),
        "R": np.diag([
            0.02,  # U-winding sensor
            0.02,  # end-bearing sensor
        ]),
        "C": np.eye(2),
    },
    "2state_B": {
        "state_cols": ["T1", "T2"],
        "meas_cols": ["T1", "T2"],
        "input_cols": ["P", "Tref"],
        "parameter_names": ["C1", "C2", "R1", "R2"],
        "theta0": np.array([
            1.5e7,  # C1
            2.5e8,  # C2
            0.011,  # R1
            0.03,   # R2
        ]),
        "lower_bound": np.array([
            1.0e4,
            1.0e4,
            0.005,
            0.005,
        ]),
        "upper_bound": np.array([
            1e8,  # C1
            5e8,  # C2
            0.5,  # R1
            2.0,  # R2
        ]),
        "Q_INIT": np.diag([
            0.0001,
            0.0001,
        ]),
        "R": np.diag([
            0.01,  # U-winding measurement
            0.001,  # End-bearing measurement
        ]),
        "C": np.eye(2),
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
Q_INIT = selected["Q_INIT"]
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

    if MODEL_OPTION == "1state":
        RENAME_MAP = {
            generic_to_raw["T1"]: "T1",      # U-winding temperature
            generic_to_raw["E1"]: "P",       # Generator power
            generic_to_raw["T16"]: "Tref",   # FAN 1 temperature
        }

    elif MODEL_OPTION == "2state_A":
        RENAME_MAP = {
            generic_to_raw["T1"]: "T1",      # U-winding temperature
            generic_to_raw["T6"]: "T2",      # LUB.OIL TEMP.
            generic_to_raw["E1"]: "P",       # Generator power
            generic_to_raw["T16"]: "Tref",   # FAN 1 temperature
        }
    elif MODEL_OPTION == "2state_B":
        RENAME_MAP = {

            generic_to_raw["T1"]: "T1",  # U-winding temperature
            generic_to_raw["T5"]: "T2",  # End-bearing temperature
            generic_to_raw["E1"]: "P",# Generator power
            generic_to_raw["T16"]: "Tref",# FAN 1 temperature
        }


SENSOR_COLS = list(RENAME_MAP.keys())