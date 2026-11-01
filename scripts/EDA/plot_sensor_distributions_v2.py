from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_matching_distributions(
    ds1_path: str | Path,
    ds2_path: str | Path,
    output_dir: str | Path,
    *,
    drop_ds1: list[str] | None = None,
    drop_ds2: list[str] | None = None,
    sensors_per_figure: int = 6,
    bins: int = 50,
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

    drop_ds1 = set(drop_ds1 or [])
    drop_ds2 = set(drop_ds2 or [])

    # ---------------------------------------------------------
    # Numeric, non-constant sensors
    # ---------------------------------------------------------
    ds1_sensors = [
        col
        for col in df1.select_dtypes(include="number").columns
        if col not in drop_ds1
        and df1[col].nunique(dropna=True) > 1
    ]

    ds2_sensors = [
        col
        for col in df2.select_dtypes(include="number").columns
        if col not in drop_ds2
        and df2[col].nunique(dropna=True) > 1
    ]

    # Canonical name -> actual column name
    map1 = {
        canonical_sensor_name(col): col
        for col in ds1_sensors
    }

    map2 = {
        canonical_sensor_name(col): col
        for col in ds2_sensors
    }

    # Sensors available in both datasets
    matching = sorted(
        set(map1.keys()) & set(map2.keys())
    )

    # Put generator power last
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
    # Plot figures
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
            figsize=(13, 2.6 * n_rows),
            squeeze=False,
        )

        # Column headings
        fig.text(
            0.30,
            0.995,
            "DS1: Port generator",
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

        fig.text(
            0.73,
            0.995,
            "DS2: Starboard generator",
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

        for row, canonical_name in enumerate(current):
            sensor1 = map1[canonical_name]
            sensor2 = map2[canonical_name]

            ax1 = axes[row, 0]
            ax2 = axes[row, 1]

            y1 = df1[sensor1].dropna()
            y2 = df2[sensor2].dropna()

            # -------------------------------------------------
            # Common x-range
            # -------------------------------------------------
            xmin = min(
                y1.min(),
                y2.min(),
            )

            xmax = max(
                y1.max(),
                y2.max(),
            )

            padding = 0.05 * (xmax - xmin)

            if padding == 0:
                padding = 1.0

            xlim = (
                xmin - padding,
                xmax + padding,
            )

            # -------------------------------------------------
            # DS1
            # -------------------------------------------------
            sns.histplot(
                y1,
                bins=bins,
                binrange=(xmin, xmax),
                stat="density",
                kde=True,
                ax=ax1,
            )

            # -------------------------------------------------
            # DS2
            # -------------------------------------------------
            sns.histplot(
                y2,
                bins=bins,
                binrange=(xmin, xmax),
                stat="density",
                kde=True,
                ax=ax2,
            )

            # Same x-axis limits
            ax1.set_xlim(xlim)
            ax2.set_xlim(xlim)

            # -------------------------------------------------
            # Same y-axis limits
            # -------------------------------------------------
            ymax = max(
                ax1.get_ylim()[1],
                ax2.get_ylim()[1],
            )

            ax1.set_ylim(0, ymax)
            ax2.set_ylim(0, ymax)

            # -------------------------------------------------
            # Statistics
            # -------------------------------------------------
            mean1 = y1.mean()
            std1 = y1.std()

            mean2 = y2.mean()
            std2 = y2.std()

            ax1.set_title(
                f"mean={mean1:.2f}, std={std1:.2f}",
                fontsize=8,
            )

            ax2.set_title(
                f"mean={mean2:.2f}, std={std2:.2f}",
                fontsize=8,
            )

            # Sensor name only once per row
            ax1.set_ylabel(
                canonical_name,
                fontsize=8,
            )

            ax2.set_ylabel("")

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

            # Only bottom row gets x-label
            if row == n_rows - 1:
                ax1.set_xlabel("Value")
                ax2.set_xlabel("Value")
            else:
                ax1.set_xlabel("")
                ax2.set_xlabel("")

        fig.tight_layout(
            rect=(0, 0, 1, 0.98)
        )

        output_path = (
            output_dir
            / f"DS1_DS2_distribution_comparison_{figure_index}.png"
        )

        fig.savefig(
            output_path,
            dpi=200,
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

    plot_matching_distributions(
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
            / "distributions"
        ),
        drop_ds1=drop_ds1,
        drop_ds2=drop_ds2,
        sensors_per_figure=6,
        bins=50,
    )


if __name__ == "__main__":
    main()