#!/usr/bin/env python3

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # No display required; save figures to files.

import matplotlib.pyplot as plt
import pandas as pd

import config as cfg


def find_time_column(
    df: pd.DataFrame,
    explicit: str | None = None,
) -> str | None:
    """
    Find the column to use for the x-axis.

    If explicit is provided, that column is used. Otherwise, the function
    searches for the first non-numeric column where more than 90% of the
    values can be parsed as timestamps.
    """
    if explicit is not None:
        if explicit not in df.columns:
            raise KeyError(
                f"Timestamp column {explicit!r} was not found."
            )

        return explicit

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue

        parsed = pd.to_datetime(
            df[column],
            errors="coerce",
            utc=True,
        )

        if parsed.notna().mean() > 0.90:
            return column

    return None


def numeric_sensor_columns(
    df: pd.DataFrame,
    time_col: str | None = None,
    drop: list[str] | None = None,
) -> list[str]:
    """
    Return all numeric columns except explicitly excluded columns.
    """
    numeric_cols = (
        df.select_dtypes(include="number")
        .columns
        .tolist()
    )

    exclude = set(drop or [])

    if time_col is not None:
        exclude.add(time_col)

    return [
        column
        for column in numeric_cols
        if column not in exclude
    ]


def make_safe_filename(name: str) -> str:
    """
    Convert a sensor name into a filename-safe string.
    """
    safe_name = "".join(
        character if character.isalnum() else "_"
        for character in name
    )

    return safe_name.strip("_")


def plot_sensor_csv(
    csv_path: str | Path,
    output_dir: str | Path,
    *,
    time_col: str | None = "Created",
    output_filename: str | None = None,
    separate: bool = False,
    drop_cols: list[str] | None = None,
    grid_cols: int = 3,
) -> None:
    """
    Plot all numeric sensors in a CSV file.

    Parameters
    ----------
    csv_path:
        Input CSV file.

    output_dir:
        Directory where plots are saved.

    time_col:
        Timestamp column. Set to None to auto-detect a timestamp column.

    output_filename:
        Filename for the combined plot. Ignored when separate=True.

    separate:
        If True, create one PNG per sensor. Otherwise, create one grid.

    drop_cols:
        Numeric columns that should not be plotted.

    grid_cols:
        Number of subplot columns in the combined figure.
    """
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(csv_path)

    selected_time_col = find_time_column(
        df,
        explicit=time_col,
    )

    if selected_time_col is not None:
        timestamps = pd.to_datetime(
            df[selected_time_col],
            errors="coerce",
            utc=True,
        )

        invalid_timestamp = timestamps.isna()

        if invalid_timestamp.any():
            n_invalid = int(invalid_timestamp.sum())

            print(
                f"Dropping {n_invalid} rows with invalid timestamps."
            )

            df = (
                df.loc[~invalid_timestamp]
                .reset_index(drop=True)
            )

            timestamps = (
                timestamps.loc[~invalid_timestamp]
                .reset_index(drop=True)
            )

        # Ensure chronological plotting.
        sort_order = timestamps.argsort()

        df = (
            df.iloc[sort_order]
            .reset_index(drop=True)
        )

        x = (
            timestamps.iloc[sort_order]
            .reset_index(drop=True)
        )

        x_label = selected_time_col

    else:
        x = df.index
        x_label = "Row index"

        print(
            "No timestamp column detected. "
            "Plotting against row index."
        )

    sensors = numeric_sensor_columns(
        df,
        time_col=selected_time_col,
        drop=drop_cols,
    )

    if not sensors:
        raise ValueError(
            f"No numeric sensor columns were found in {csv_path}."
        )

    print(f"Plotting {len(sensors)} sensors from: {csv_path}")

    base_name = csv_path.stem

    # ---------------------------------------------------------
    # One figure per sensor
    # ---------------------------------------------------------
    if separate:
        sensor_output_dir = output_dir / base_name

        sensor_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for sensor in sensors:
            fig, ax = plt.subplots(
                figsize=(10, 3.5)
            )

            ax.plot(
                x,
                df[sensor],
                linewidth=0.8,
            )

            ax.set_title(sensor)
            ax.set_xlabel(x_label)
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

            fig.autofmt_xdate()
            fig.tight_layout()

            safe_name = make_safe_filename(sensor)

            output_path = (
                sensor_output_dir
                / f"{base_name}_{safe_name}.png"
            )

            fig.savefig(
                output_path,
                dpi=120,
            )

            plt.close(fig)

        print(
            f"Saved {len(sensors)} plots to: "
            f"{sensor_output_dir}"
        )

        return

    # ---------------------------------------------------------
    # Combined subplot grid
    # ---------------------------------------------------------
    n_sensors = len(sensors)
    n_cols = max(1, grid_cols)
    n_rows = math.ceil(n_sensors / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            n_cols * 4.5,
            n_rows * 2.6,
        ),
        squeeze=False,
    )

    for index, sensor in enumerate(sensors):
        row = index // n_cols
        column = index % n_cols

        ax = axes[row, column]

        ax.plot(
            x,
            df[sensor],
            linewidth=0.7,
        )

        ax.set_title(
            sensor,
            fontsize=8,
        )

        ax.tick_params(
            labelsize=6,
        )

        ax.grid(
            True,
            alpha=0.3,
        )

    # Hide unused subplot panels.
    for index in range(
        n_sensors,
        n_rows * n_cols,
    ):
        row = index // n_cols
        column = index % n_cols

        axes[row, column].axis("off")

    fig.suptitle(
        f"{base_name} — {n_sensors} sensors",
        fontsize=12,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.98),
    )

    if output_filename is None:
        output_filename = f"{base_name}_sensors.png"

    output_path = output_dir / output_filename

    fig.savefig(
        output_path,
        dpi=130,
    )

    plt.close(fig)

    print(f"Saved combined plot to: {output_path}")


def main():
    dataset = 1
    dataset_name = f"DS{dataset}"

    csv_path = (
        cfg.DATA_PATH
        / "subsystems"
        / dataset_name
        / "NUMERIC"
        / "AE_PORT.csv"
    )

    output_dir = (
        cfg.DATA_PATH
        / "EDA"
        / dataset_name
        / "AE_PORT"
    )

    plot_sensor_csv(
        csv_path=csv_path,
        output_dir=output_dir,

        # Use "Created", or None to auto-detect.
        time_col="Created",

        # False gives one combined grid.
        # True gives one PNG per sensor.
        separate=False,

        # Numeric columns that should not be plotted.
        drop_cols=[
            "POWER_kW_sq",
            "AE PS RUNNING",
            "AE PS POWER COUNTER",
        ],

        # Number of columns in the combined grid.
        grid_cols=3,

        output_filename=(
            f"{dataset_name}_AE_PORT_sensor_plots.png"
        ),
    )

    dataset = 2
    dataset_name = f"DS{dataset}"

    csv_path = (
            cfg.DATA_PATH
            / "subsystems"
            / dataset_name
            / "NUMERIC"
            / "AE_STBD.csv"
    )

    output_dir = (
            cfg.DATA_PATH
            / "EDA"
            / dataset_name
            / "AE_STBD"
    )

    plot_sensor_csv(
        csv_path=csv_path,
        output_dir=output_dir,

        # Use "Created", or None to auto-detect.
        time_col="Created",

        # False gives one combined grid.
        # True gives one PNG per sensor.
        separate=True,

        # Numeric columns that should not be plotted.
        drop_cols=[
            "POWER_kW_sq",
            "AE SB RUNNING",
            "AE SB POWER COUNTER",
        ],

        # Number of columns in the combined grid.
        grid_cols=3,

        output_filename=(
            f"{dataset_name}_AE_STBD_sensor_plots.png"
        ),
    )



if __name__ == "__main__":
    main()