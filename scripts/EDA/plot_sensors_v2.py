from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd
import config as cfg

def canonical_sensor_name(name: str) -> str:
    name = name.strip()

    replacements = [
        ("AE PORT ", ""),
        ("AE STBD ", ""),
        ("AE_PS_", ""),
        ("AE_SB_", ""),
        ("AE PS ", ""),
        ("AE SB ", ""),
    ]

    for old, new in replacements:
        name = name.replace(old, new)

    return name.strip()


def plot_matching_sensors(
    ds1_path: str | Path,
    ds2_path: str | Path,
    output_dir: str | Path,
    *,
    time_col: str = "Created",
    drop_ds1: list[str] | None = None,
    drop_ds2: list[str] | None = None,
    sensors_per_figure: int = 6,
):
    ds1_path = Path(ds1_path)
    ds2_path = Path(ds2_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------
    df1 = pd.read_csv(ds1_path)
    df2 = pd.read_csv(ds2_path)

    df1[time_col] = pd.to_datetime(
        df1[time_col],
        errors="coerce",
        utc=True,
    )

    df2[time_col] = pd.to_datetime(
        df2[time_col],
        errors="coerce",
        utc=True,
    )

    df1 = (
        df1.dropna(subset=[time_col])
        .sort_values(time_col)
        .reset_index(drop=True)
    )

    df2 = (
        df2.dropna(subset=[time_col])
        .sort_values(time_col)
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Get numeric sensors
    # ---------------------------------------------------------
    drop_ds1 = set(drop_ds1 or [])
    drop_ds2 = set(drop_ds2 or [])

    ds1_sensors = [
        col
        for col in df1.select_dtypes(include="number").columns
        if col not in drop_ds1
    ]

    ds2_sensors = [
        col
        for col in df2.select_dtypes(include="number").columns
        if col not in drop_ds2
    ]

    # canonical name -> actual column name
    map1 = {
        canonical_sensor_name(col): col
        for col in ds1_sensors
    }

    map2 = {
        canonical_sensor_name(col): col
        for col in ds2_sensors
    }

    # Only sensors available in both datasets
    matching = sorted(
        set(map1.keys()) & set(map2.keys())
    )
    # Put generator power last so it appears as the bottom plot
    if "POWER_kW" in matching:
        matching.remove("POWER_kW")
        matching.append("POWER_kW")

    print(f"Matching sensors: {len(matching)}")

    for name in matching:
        print(
            f"{name:35s} "
            f"DS1={map1[name]!r} | "
            f"DS2={map2[name]!r}"
        )

    # ---------------------------------------------------------
    # Split into manageable figures
    # ---------------------------------------------------------
    for figure_index, start in enumerate(
        range(0, len(matching), sensors_per_figure),
        start=1,
    ):
        current = matching[
            start:start + sensors_per_figure
        ]

        n_rows = len(current)

        fig, axes = plt.subplots(
            n_rows,
            2,
            figsize=(13, 2.5 * n_rows),
            squeeze=False,
        )

        # Column headings
        axes[0, 0].set_title(
            "DS1: Port generator",
            fontsize=11,
            fontweight="bold",
        )

        axes[0, 1].set_title(
            "DS2: Starboard generator",
            fontsize=11,
            fontweight="bold",
        )

        for row, canonical_name in enumerate(current):
            sensor1 = map1[canonical_name]
            sensor2 = map2[canonical_name]

            ax1 = axes[row, 0]
            ax2 = axes[row, 1]

            y1 = df1[sensor1]
            y2 = df2[sensor2]

            ax1.plot(
                df1[time_col],
                y1,
                linewidth=0.7,
            )

            ax2.plot(
                df2[time_col],
                y2,
                linewidth=0.7,
            )

            # -------------------------------------------------
            # Same y limits for corresponding sensors
            # -------------------------------------------------
            ymin = min(
                y1.min(skipna=True),
                y2.min(skipna=True),
            )

            ymax = max(
                y1.max(skipna=True),
                y2.max(skipna=True),
            )

            padding = 0.05 * (ymax - ymin)

            if padding == 0:
                padding = 1.0

            ax1.set_ylim(
                ymin - padding,
                ymax + padding,
            )

            ax2.set_ylim(
                ymin - padding,
                ymax + padding,
            )

            # Sensor name shown once for the row
            ax1.set_ylabel(
                canonical_name,
                fontsize=8,
            )

            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)

            ax1.tick_params(
                axis="both",
                labelsize=7,
            )

            ax2.tick_params(
                axis="both",
                labelsize=7,
            )

            # Only bottom row needs x-axis label
            if row == n_rows - 1:
                ax1.set_xlabel("Time")
                ax2.set_xlabel("Time")

        fig.autofmt_xdate()

        fig.tight_layout(
            rect=(0, 0, 1, 0.98)
        )

        output_path = (
            output_dir
            / f"DS1_DS2_sensor_comparison_{figure_index}.png"
        )

        fig.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(f"Saved: {output_path}")



def main():
    drop_ds1 = [
        "POWER_kW_sq",
        "AE PS RUNNING",
        "AE PS POWER COUNTER",
        "AE PORT CYL.1 EXH.GAS TEMP. DEV",
        "AE PORT CYL.2 EXH.GAS TEMP. DEV",
        "AE PORT CYL.3 EXH.GAS TEMP. DEV",
        "AE PORT CYL.4 EXH.GAS TEMP. DEV",
        "AE PORT CYL.5 EXH.GAS TEMP. DEV",
        "AE PORT CYL.6 EXH.GAS TEMP. DEV",
    ]

    drop_ds2 = [
        "POWER_kW_sq",
        "AE SB RUNNING",
        "AE SB POWER COUNTER",
        "AE STBD CYL.1 EXH.GAS TEMP. DEV",
        "AE STBD CYL.2 EXH.GAS TEMP. DEV",
        "AE STBD CYL.3 EXH.GAS TEMP. DEV",
        "AE STBD CYL.4 EXH.GAS TEMP. DEV",
        "AE STBD CYL.5 EXH.GAS TEMP. DEV",
        "AE STBD CYL.6 EXH.GAS TEMP. DEV",
    ]

    plot_matching_sensors(
        ds1_path=(
                cfg.DATA_PATH
                / "subsystems"
                / "DS1"
                / "NUMERIC"
                / "AE_PORT.csv"
        ),
        ds2_path=(
                cfg.DATA_PATH
                / "subsystems"
                / "DS2"
                / "NUMERIC"
                / "AE_STBD.csv"
        ),
        output_dir=(
                cfg.DATA_PATH
                / "EDA"
                / "DS1_DS2_comparison"
        ),
        drop_ds1=drop_ds1,
        drop_ds2=drop_ds2,

        # Six sensor pairs per figure
        sensors_per_figure=6,
    )
if __name__ == "__main__":
    main()