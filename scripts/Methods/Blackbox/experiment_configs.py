import torch

# ---------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
SEED = 42
# ---------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------
TS_COL = "Created"

WINDOW_STEPS = 300
VAL_PROFILE_LEN = 3
TEST_PROFILE_LEN = 3

SENSOR_COLS = [
    "AE PORT GEN.U-WINDING TEMP.",
    "AE PORT GEN.V-WINDING TEMP.",
    "AE PORT GEN.W-WINDING TEMP.",
    "AE_PS_EXH",
    "AE PORT END BRG.TEMP.",
    "AE PORT LUB.OIL TEMP.",
    "AE PORT HT FW OUTLET TEMP.",
    "POWER_kW",
]

RENAME_MAP = {
    "AE PORT GEN.U-WINDING TEMP.": "T1",
    "AE PORT GEN.V-WINDING TEMP.": "T2",
    "AE PORT GEN.W-WINDING TEMP.": "T3",
    "AE_PS_EXH": "T4",
    "AE PORT END BRG.TEMP.": "T5",
    "AE PORT LUB.OIL TEMP.": "T6",
    "AE PORT HT FW OUTLET TEMP.": "T7",
    "POWER_kW": "E1",
}

INPUT_COLS = [
    "E1",
    "T4",
    "T5",
    "T6",
    "T7",
]

TARGET_COLS = [
    "T1",
    "T2",
    "T3",
]

# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------
MODEL_TYPE = "TCN"

MODEL_CONFIG = {
    "input_size": len(INPUT_COLS),
    "output_size": len(TARGET_COLS),
    "channels": (32, 64, 64, 64, 64),
    "kernel_size": 3,
    "dropout": 0.1,
}

# ---------------------------------------------------------
# Training configuration
# ---------------------------------------------------------
TRAINING_CONFIG = {
    "n_epochs": 200,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "max_grad_norm": 1.0,
}