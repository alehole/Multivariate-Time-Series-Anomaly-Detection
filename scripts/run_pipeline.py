from scripts.Subsystem_grouping.categorize_raw_dataset import main as categorize_raw_dataset
from scripts.Subsystem_grouping.group_categorized_dataset import main as group_categorized_dataset

from scripts.misc.combine_csv import main as combine_csv
from scripts.misc.feature_engineering import main as feature_engineering
from scripts.misc.train_val_test_split import main as train_val_test_split

from scripts.EDA.auto_correlation import main as auto_correlation
from scripts.EDA.compare_sensors import main as compare_sensors
from scripts.EDA.correlation_analysis import main as correlation_analysis
from scripts.EDA.cross_correlation import main as cross_correlation
from scripts.EDA.plot_sensor_distributions import main as plot_distributions
from scripts.EDA.plot_sensor_distributions_v2 import main as plot_distributions_v2
from scripts.EDA.plot_sensors import main as plot_sensors
from scripts.EDA.plot_sensors_v2 import main as plot_sensors_v2
from scripts.EDA.sea_state import main as sea_state
from scripts.EDA.sensor_statistics import main as sensor_statistics
from scripts.EDA.temperature_API import main as temperature

from scripts.Visualization.plotly_plot import main as plotly_plot
from scripts.Visualization.plot_vessel_loc import main as vessel_loc



PIPELINE = [
    # 1. Dataset grouping
    ("Categorize raw dataset", categorize_raw_dataset, True),
    ("Group categorized dataset", group_categorized_dataset, True),

    # 2. Data preprocessing and feature engineering
    ("Combine CSV", combine_csv, True),
    ("Feature engineering", feature_engineering, True),
    ("Train/Val/Test split", train_val_test_split, True),

    # 3. Exploratory Data Analysis
    ("Sensor statistics", sensor_statistics, False),
    ("Compare sensors", compare_sensors, False),
    ("Correlation analysis", correlation_analysis, False),
    ("Auto correlation", auto_correlation, False),
    ("Cross correlation", cross_correlation, False),
    ("Plot distributions", plot_distributions, False),
    ("Plot distributions v2", plot_distributions_v2, True),
    ("Plot sensors", plot_sensors, False),
    ("Plot sensors v2", plot_sensors_v2, False),
    ("Sea state", sea_state, False),
    ("Temperature", temperature, False),

    # 4. Visualization
    ("Plotly plot", plotly_plot, False),
    ("Vessel location", vessel_loc, False),
]


def main():
    for name, function, enabled in PIPELINE:
        if not enabled:
            print(f"[SKIPPED] {name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Running: {name}")
        print(f"{'=' * 60}", flush=True)

        try:
            function()
            print(f"[COMPLETED] {name}")

        except Exception as e:
            print(f"[FAILED] {name}: {e}")
            raise

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()
