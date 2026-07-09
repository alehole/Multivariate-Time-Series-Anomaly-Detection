from pathlib import Path
import platform

OS_NAME = platform.system()

BASE_PROJ_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_PROJ_DIR.parent / "data"
DS1_RAW = DATA_PATH /"raw" / "DS1"/"LiveData.csv"
DS2_RAW = DATA_PATH /"raw" / "DS2"/"LiveData.csv"

DS1_CATEGORIZED_DIR = DATA_PATH / "raw_categorized" / "DS1"
DS2_CATEGORIZED_DIR = DATA_PATH / "raw_categorized" / "DS2"