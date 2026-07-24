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

# ---------------------------------------------------------
# Data model configuration shared by all methods
# ---------------------------------------------------------

TS_COL = "Created"

SENSOR_COLS = [
    "AE PORT GEN.U-WINDING TEMP.",
    "AE PORT GEN.V-WINDING TEMP.",
    "AE PORT GEN.W-WINDING TEMP.",
    "AE_PS_EXH",
    "AE PORT END BRG.TEMP.",
    "AE PORT LUB.OIL TEMP.",
    "AE PORT HT FW OUTLET TEMP.",
    "AE PORT TC EXH.GAS OUT.TEMP.",
    "AE PORT LUB.OIL PRESS.",
    "POWER_kW",
    "POWER_kW_sq",
    "AE PORT CYL.1 EXH.GAS TEMP.",
    "AE PORT CYL.2 EXH.GAS TEMP.",
    "AE PORT CYL.3 EXH.GAS TEMP.",
    "AE PORT CYL.4 EXH.GAS TEMP.",
    "AE PORT CYL.5 EXH.GAS TEMP.",
    "AE PORT CYL.6 EXH.GAS TEMP.",
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
    "P1",
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
# CONFIGURATIONS MC1-MC4
# ---------------------------------------------------------
CONFIG = "MC2"

if CONFIG == "MC1":
    INPUT_COLS = [
        "E1", # Power
    ]
    TARGET_COLS = [
        "T1",
        "T2",
        "T3",
    ]
elif CONFIG == "MC2":
    INPUT_COLS = [
        "E1", # Power
        "T4", # Exhaust
    ]
    TARGET_COLS = [
        "T1",
        "T2",
        "T3",
    ]
