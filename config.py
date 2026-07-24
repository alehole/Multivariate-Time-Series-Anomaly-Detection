from pathlib import Path
import platform
import torch

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
OS_NAME = platform.system()

BASE_PROJ_DIR = Path(__file__).resolve()
DATA_PATH = BASE_PROJ_DIR.parent / "data"

DS1_RAW = DATA_PATH /"raw" / "DS1"/"LiveData.csv"
DS2_RAW = DATA_PATH /"raw" / "DS2"/"LiveData.csv"

DS1_CATEGORIZED_DIR = DATA_PATH / "raw_categorized" / "DS1"
DS2_CATEGORIZED_DIR = DATA_PATH / "raw_categorized" / "DS2"


# ---------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
SEED = 42
TS_COL = "Created"

# ---------------------------------------------------------
# Data model configuration shared by all methods
# ---------------------------------------------------------
RENAME_MAP = {
    "AE PORT GEN.U-WINDING TEMP.": "T1",
    "AE PORT GEN.V-WINDING TEMP.": "T2",
    "AE PORT GEN.W-WINDING TEMP.": "T3",
    "AE_PS_EXH": "T4",
    "AE PORT END BRG.TEMP.": "T5",
    "AE PORT LUB.OIL TEMP.": "T6",
    "AE PORT HT FW OUTLET TEMP.": "T7",
    "AE PORT TC EXH.GAS OUT.TEMP.": "T8",
    "AE PORT LUB.OIL PRESS.": "P1",
    "POWER_kW": "E1",
    "POWER_kW_sq": "E2",
    "AE PORT CYL.1 EXH.GAS TEMP.": "T9",
    "AE PORT CYL.2 EXH.GAS TEMP.": "T10",
    "AE PORT CYL.3 EXH.GAS TEMP.": "T11",
    "AE PORT CYL.4 EXH.GAS TEMP.": "T12",
    "AE PORT CYL.5 EXH.GAS TEMP.": "T13",
    "AE PORT CYL.6 EXH.GAS TEMP.": "T14",
}

# ---------------------------------------------------------
# Global model configurations
# ---------------------------------------------------------
CONFIG = "MC2"

MODEL_CONFIGS = {
    "MC1": {
        "input_cols": ["E1"],
        "target_cols": ["T1", "T2", "T3"],
    },
    "MC2": {
        "input_cols": ["E1", "T4"],
        "target_cols": ["T1", "T2", "T3"],
    },
    "MC3": {
        "input_cols": ["E1","T4" ,"T6"],
        "target_cols": ["T1", "T2", "T3"],
    },
    "MC4": {
        "input_cols": ["E1", "T4", "T5", "T6", "T7"],
        "target_cols": ["T1", "T2", "T3"],
    },
}

INPUT_COLS = MODEL_CONFIGS[CONFIG]["input_cols"]
TARGET_COLS = MODEL_CONFIGS[CONFIG]["target_cols"]
SENSOR_COLS = list(RENAME_MAP.keys())