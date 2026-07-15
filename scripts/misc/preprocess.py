from pathlib import Path
import shutil

import config as cfg
from scripts.Subsystem_grouping import group_categorized_dataset, categorize_raw_dataset
from scripts.misc import combine_csv, feature_engineering
from scripts.EDA import (
    sea_state,
    plot_sensor_distributions,
    sensor_statistics,
    temperature_API,
    correlation_analysis,
    auto_correlation,
    cross_correlation,
    plot_sensors,
)


def delete_directory(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
        print(f"Deleted: {directory}")
    else:
        print(f"Not found, skipped: {directory}")


def main():
    directories_to_delete = [
        cfg.DATA_PATH / "EDA",
        cfg.DATA_PATH / "raw_categorized",
        cfg.DATA_PATH / "subsystems",
    ]

    for directory in directories_to_delete:
        delete_directory(directory)

    ## subsystem grouping
    categorize_raw_dataset.main()
    group_categorized_dataset.main()

    #misc
    combine_csv.main()
    feature_engineering.main()

    #EDA
    temperature_API.main()
    sea_state.main()
    plot_sensor_distributions.main()
    sensor_statistics.main()
    correlation_analysis.main()
    auto_correlation.main()
    cross_correlation.main()
    plot_sensors.main()



if __name__ == "__main__":
    main()