import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg

def plot_acf_overlay_grid(
    acf_original: pd.DataFrame,
    acf_differenced: pd.DataFrame,
    output_path: str | Path,
    *,
    dt_s: float = np.nan,
    title: str = "Autocorrelation functions",
    ncols: int = 3,
    xmax_minutes: float | None = None,
    show_one_over_e: bool = True,
) -> None:
    """
    Plot the original and differenced ACF curves overlaid, one panel per
    sensor. Both frames must be indexed by lag in samples; they may use
    different maximum lags.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sensors = [
        column
        for column in acf_original.columns
        if column in acf_differenced.columns
    ]

    if not sensors:
        print(f"[SKIP] No shared sensors for {output_path}")
        return

    n_sensors = len(sensors)
    ncols = min(ncols, n_sensors)
    nrows = math.ceil(n_sensors / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            ncols * 5.0,
            nrows * 3.2,
        ),
        squeeze=False,
    )

    axes_flat = axes.flatten()

    def lag_axis(index: pd.Index) -> tuple[np.ndarray, str]:
        lag_samples = index.to_numpy()
        if np.isfinite(dt_s):
            return lag_samples * dt_s / 60.0, "Lag [min]"
        return lag_samples, "Lag [samples]"

    x_original, x_label = lag_axis(acf_original.index)
    x_differenced, _ = lag_axis(acf_differenced.index)

    for index, sensor in enumerate(sensors):
        ax = axes_flat[index]

        ax.axhline(
            0.0,
            color="0.6",
            linewidth=0.8,
        )

        if show_one_over_e:
            ax.axhline(
                1.0 / np.e,
                color="0.75",
                linewidth=0.8,
                linestyle=":",
            )

        ax.plot(
            x_original,
            acf_original[sensor].to_numpy(),
            linewidth=1.4,
            label="original",
        )

        ax.plot(
            x_differenced,
            acf_differenced[sensor].to_numpy(),
            linewidth=1.4,
            linestyle="--",
            label="differenced",
        )

        if xmax_minutes is not None:
            ax.set_xlim(0.0, xmax_minutes)

        ax.set_ylim(-0.75, 1.05)
        ax.set_title(sensor, fontsize=8)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Autocorrelation")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    # Single legend on the first panel.
    axes_flat[0].legend(
        fontsize=8,
        loc="upper right",
        framealpha=0.9,
    )

    # Hide unused subplot positions.
    for index in range(n_sensors, len(axes_flat)):
        axes_flat[index].axis("off")

    fig.suptitle(
        title,
        fontsize=14,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.98)
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[OK] Saved overlay ACF plot: {output_path}")


def run_acf_overlay(
    df: pd.DataFrame,
    out_dir: str | Path,
    tag: str,
    *,
    time_col: str = "Created",
    max_lag_original: int = 360,
    max_lag_differenced: int = 60,
    ncols: int = 3,
    xmax_minutes: float | None = 60.0,
    drop_cols: list[str] | None = None,
) -> None:
    """
    Compute the original and first-differenced ACF for each sensor and
    plot them overlaid on shared axes.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    drop_cols = drop_cols or []

    numeric_df = (
        df.select_dtypes(include="number")
        .drop(columns=drop_cols, errors="ignore")
        .copy()
    )

    # Remove constant columns.
    numeric_df = numeric_df.loc[
        :,
        numeric_df.nunique(dropna=True) > 1
    ]

    if numeric_df.shape[1] == 0:
        print(f"[SKIP] {tag}: no usable numeric columns.")
        return

    dt_s = get_sampling_interval(
        df,
        time_col=time_col,
    )

    columns = numeric_df.columns.tolist()

    acf_original = calculate_acf_table(
        df=numeric_df,
        columns=columns,
        max_lag=max_lag_original,
    )

    acf_differenced = calculate_acf_table(
        df=numeric_df.diff(),
        columns=columns,
        max_lag=max_lag_differenced,
    )

    plot_acf_overlay_grid(
        acf_original=acf_original,
        acf_differenced=acf_differenced,
        output_path=(
            out_dir
            / f"autocorrelation_overlay_{tag}.png"
        ),
        dt_s=dt_s,
        title=f"{tag}: original vs. differenced autocorrelation",
        ncols=ncols,
        xmax_minutes=xmax_minutes,
    )
def get_sampling_interval(
    df: pd.DataFrame,
    time_col: str = "Created",
) -> float:
    """
    Estimate the median sampling interval in seconds.
    """
    if time_col not in df.columns:
        return np.nan

    timestamps = pd.to_datetime(
        df[time_col],
        errors="coerce",
        utc=True,
    )

    dt_s = (
        timestamps
        .dropna()
        .sort_values()
        .diff()
        .dt.total_seconds()
        .dropna()
    )

    dt_s = dt_s[dt_s > 0]

    if dt_s.empty:
        return np.nan

    return float(dt_s.median())


def calculate_acf(
    series: pd.Series,
    max_lag: int,
) -> pd.Series:
    """
    Calculate the autocorrelation function for one signal.

    Lag k represents corr(x_t, x_{t-k}).

    Missing values are handled pairwise and are not removed before
    shifting, which preserves their original positions in time.
    """
    series = pd.to_numeric(
        series,
        errors="coerce",
    )

    acf_values = {}

    for lag in range(max_lag + 1):
        if lag == 0:
            acf_values[lag] = 1.0
        else:
            acf_values[lag] = series.corr(
                series.shift(lag)
            )

    return pd.Series(
        acf_values,
        name=series.name,
        dtype=float,
    )


def calculate_acf_table(
    df: pd.DataFrame,
    columns: list[str],
    max_lag: int,
) -> pd.DataFrame:
    """
    Calculate ACF values for all selected sensors.

    Returns a DataFrame where:
        rows = lags
        columns = sensors
    """
    acf_data = {
        column: calculate_acf(
            df[column],
            max_lag=max_lag,
        )
        for column in columns
    }

    acf_df = pd.DataFrame(acf_data)
    acf_df.index.name = "lag_samples"

    return acf_df


def create_acf_summary(
    acf_df: pd.DataFrame,
    dt_s: float,
    selected_lags: tuple[int, ...] = (1, 5, 10, 30, 60),
) -> pd.DataFrame:
    """
    Create a compact summary of the ACF results.
    """
    rows = []

    for sensor in acf_df.columns:
        acf = acf_df[sensor]

        # Exclude lag zero because its autocorrelation is always 1.
        nonzero_acf = acf.iloc[1:].dropna()

        if nonzero_acf.empty:
            max_abs_acf = np.nan
            max_abs_lag = np.nan
            max_abs_lag_minutes = np.nan
        else:
            max_abs_lag = int(nonzero_acf.abs().idxmax())
            max_abs_acf = float(nonzero_acf.loc[max_abs_lag])

            if np.isfinite(dt_s):
                max_abs_lag_minutes = (
                    max_abs_lag * dt_s / 60.0
                )
            else:
                max_abs_lag_minutes = np.nan

        row = {
            "sensor": sensor,
            "max_abs_acf": max_abs_acf,
            "max_abs_acf_lag_samples": max_abs_lag,
            "max_abs_acf_lag_minutes": max_abs_lag_minutes,
        }

        for lag in selected_lags:
            column_name = f"acf_lag_{lag}"

            if lag in acf.index:
                row[column_name] = acf.loc[lag]
            else:
                row[column_name] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def plot_acf_grid(
    acf_df: pd.DataFrame,
    output_path: str | Path,
    *,
    dt_s: float = np.nan,
    title: str = "Autocorrelation functions",
    ncols: int = 3,
    confidence_limit: float | None = None,
) -> None:
    """
    Plot all sensor ACF curves in one figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sensors = list(acf_df.columns)

    if not sensors:
        print(f"[SKIP] No sensors available for {output_path}")
        return

    n_sensors = len(sensors)
    ncols = min(ncols, n_sensors)
    nrows = math.ceil(n_sensors / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(
            ncols * 5.0,
            nrows * 3.2,
        ),
        squeeze=False,
    )

    axes_flat = axes.flatten()
    lag_samples = acf_df.index.to_numpy()

    if np.isfinite(dt_s):
        x_values = lag_samples * dt_s / 60.0
        x_label = "Lag [min]"
    else:
        x_values = lag_samples
        x_label = "Lag [samples]"

    for index, sensor in enumerate(sensors):
        ax = axes_flat[index]

        values = acf_df[sensor].to_numpy()

        ax.plot(
            x_values,
            values,
            linewidth=1.0,
        )

        ax.axhline(
            0.0,
            linewidth=0.8,
        )

        if confidence_limit is not None:
            ax.axhline(
                confidence_limit,
                linestyle="--",
                linewidth=0.8,
            )

            ax.axhline(
                -confidence_limit,
                linestyle="--",
                linewidth=0.8,
            )

        ax.set_ylim(-1.05, 1.05)
        ax.set_title(sensor, fontsize=8)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Autocorrelation")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    # Hide unused subplot positions.
    for index in range(n_sensors, len(axes_flat)):
        axes_flat[index].axis("off")

    fig.suptitle(
        title,
        fontsize=14,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.98)
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"[OK] Saved ACF plot: {output_path}")


def run_acf(
    df: pd.DataFrame,
    out_dir: str | Path,
    tag: str,
    *,
    time_col: str = "Created",
    max_lag: int = 120,
    ncols: int = 3,
    difference: bool = False,
    drop_cols: list[str] | None = None,
) -> pd.DataFrame:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    drop_cols = drop_cols or []

    numeric_df = (
        df.select_dtypes(include="number")
        .drop(columns=drop_cols, errors="ignore")
        .copy()
    )

    # Remove constant columns.
    numeric_df = numeric_df.loc[
        :,
        numeric_df.nunique(dropna=True) > 1
    ]

    if numeric_df.shape[1] == 0:
        print(f"[SKIP] {tag}: no usable numeric columns.")
        return pd.DataFrame()

    if difference:
        numeric_df = numeric_df.diff()
        analysis_name = "differenced"
    else:
        analysis_name = "original"

    dt_s = get_sampling_interval(
        df,
        time_col=time_col,
    )

    acf_df = calculate_acf_table(
        df=numeric_df,
        columns=numeric_df.columns.tolist(),
        max_lag=max_lag,
    )

    summary_df = create_acf_summary(
        acf_df=acf_df,
        dt_s=dt_s,
    )

    acf_df.to_csv(
        out_dir / f"autocorrelation_{analysis_name}_{tag}.csv"
    )

    summary_df.to_csv(
        out_dir / f"autocorrelation_summary_{analysis_name}_{tag}.csv",
        index=False,
    )

    n_rows = len(numeric_df)

    confidence_limit = (
        1.96 / np.sqrt(n_rows)
        if n_rows > 0
        else None
    )

    plot_acf_grid(
        acf_df=acf_df,
        output_path=(
            out_dir
            / f"autocorrelation_{analysis_name}_{tag}.png"
        ),
        dt_s=dt_s,
        title=f"{tag}: {analysis_name} autocorrelation",
        ncols=ncols,
        confidence_limit=confidence_limit,
    )

    print(
        f"[{tag}] sensors={acf_df.shape[1]}, "
        f"maximum lag={max_lag} samples, "
        f"median dt={dt_s:.2f} s"
    )

    return acf_df


def loop_folder(
    base_dir: Path,
    out_root: Path,
    *,
    time_col: str = "Created",
    max_lag: int = 120,
    ncols: int = 3,
    difference: bool = False,
    drop_cols: list[str] | None = None,
) -> None:

    base_dir = Path(base_dir)
    out_root = Path(out_root)

    if not base_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {base_dir}"
        )

    csv_files = sorted(base_dir.glob("*.csv"))

    if not csv_files:
        print(f"[WARNING] No CSV files found in {base_dir}")
        return

    for csv_path in csv_files:
        tag = csv_path.stem
        out_dir = out_root / tag

        print(f"\nProcessing ACF: {csv_path}")

        df = pd.read_csv(csv_path)

        run_acf(
            df=df,
            out_dir=out_dir,
            tag=tag,
            time_col=time_col,
            max_lag=max_lag,
            ncols=ncols,
            difference=difference,
            drop_cols=drop_cols,
        )


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
        /"autocorrelation"
    )

    # NOTE: no trailing comma — otherwise drop_cols becomes a 1-tuple
    # containing the list, and the .drop(columns=...) call misbehaves.
    drop_cols = [
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
    df = pd.read_csv(csv_path)

    # Original-signal ACF: shows the (long) load-driven memory.
    run_acf(
        df=df,
        out_dir=output_dir,
        tag=f"{dataset_name}_AE_PORT",
        max_lag=240,
        difference=False,
        drop_cols=drop_cols,
    )

    # Differenced ACF: removes the slow load trend so the
    # short-term / residual structure is visible.
    run_acf(
        df=df,
        out_dir=output_dir,
        tag=f"{dataset_name}_AE_PORT",
        max_lag=60,
        difference=True,
        drop_cols=drop_cols,
    )
    # Overlay of original and differenced ACF on shared axes.
    run_acf_overlay(
        df=df,
        out_dir=output_dir,
        tag=f"{dataset_name}_AE_PORT",
        max_lag_original=240,
        max_lag_differenced=240,
        xmax_minutes=None,
        drop_cols=drop_cols,
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
        /"autocorrelation"
    )

    # NOTE: no trailing comma — otherwise drop_cols becomes a 1-tuple
    # containing the list, and the .drop(columns=...) call misbehaves.
    drop_cols = [
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

    df = pd.read_csv(csv_path)

    # Original-signal ACF: shows the (long) load-driven memory.
    run_acf(
        df=df,
        out_dir=output_dir,
        tag=f"{dataset_name}_AE_STBD",
        max_lag=240,
        difference=False,
        drop_cols=drop_cols,
    )

    # Differenced ACF: removes the slow load trend so the
    # short-term / residual structure is visible.
    run_acf(
        df=df,
        out_dir=output_dir,
        tag=f"{dataset_name}_AE_STBD",
        max_lag=60,
        difference=True,
        drop_cols=drop_cols,
    )
    # Overlay of original and differenced ACF on shared axes.
    run_acf_overlay(
        df=df,
        out_dir=output_dir,
        tag=f"{dataset_name}_AE_STBD",
        max_lag_original=240,
        max_lag_differenced=240,
        xmax_minutes=None,
        drop_cols=drop_cols,
    )

if __name__ == "__main__":
    main()