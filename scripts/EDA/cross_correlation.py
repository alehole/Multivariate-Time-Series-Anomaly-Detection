import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg


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


def ccf_pair(
    ref: pd.Series,
    sig: pd.Series,
    max_lag: int,
    *,
    difference: bool = False,
) -> pd.Series:
    """
    Cross-correlation between a reference and a signal over a range of lags.

    Sign convention: value at lag k is corr(ref_t, sig_{t+k}). A peak at a
    positive lag therefore means the reference *leads* the signal (the signal
    responds later); a peak at a negative lag means the reference lags it.

    Index is the lag in samples, running from -max_lag to +max_lag.
    """
    ref = pd.to_numeric(ref, errors="coerce")
    sig = pd.to_numeric(sig, errors="coerce")

    if difference:
        ref = ref.diff()
        sig = sig.diff()

    values = {
        lag: ref.corr(sig.shift(-lag))
        for lag in range(-max_lag, max_lag + 1)
    }

    series = pd.Series(values, dtype=float)
    series.index.name = "lag_samples"
    return series


def _peak(ccf: pd.Series) -> tuple[float, int]:
    """Return (signed peak correlation, lag at peak) using the largest |corr|."""
    valid = ccf.dropna()
    if valid.empty:
        return np.nan, 0
    lag = int(valid.abs().idxmax())
    return float(valid.loc[lag]), lag


def ccf_all_pairs_long(
    x: pd.DataFrame,
    max_lag: int,
    *,
    dt_s: float = np.nan,
    difference: bool = False,
) -> pd.DataFrame:
    """
    Cross-correlation for every unordered sensor pair.

    For each pair (sensor_1, sensor_2) the lag convention is that a positive
    best_lag means sensor_1 leads sensor_2. Returns one row per pair with the
    zero-lag correlation, the peak correlation, and the lag at which it occurs.
    """
    columns = x.columns.tolist()
    rows = []

    for a, b in itertools.combinations(columns, 2):
        ccf = ccf_pair(x[a], x[b], max_lag, difference=difference)
        peak_corr, best_lag = _peak(ccf)
        zero_lag = ccf.get(0, np.nan)

        best_lag_minutes = (
            best_lag * dt_s / 60.0
            if np.isfinite(dt_s)
            else np.nan
        )

        rows.append(
            {
                "sensor_1": a,
                "sensor_2": b,
                "zero_lag_corr": zero_lag,
                "peak_corr": peak_corr,
                "best_lag_samples": best_lag,
                "best_lag_minutes": best_lag_minutes,
            }
        )

    return pd.DataFrame(rows)


def plot_lag_heatmap(
    long_df: pd.DataFrame,
    columns: list[str],
    *,
    save_path: str | Path | None = None,
    dt_s: float = np.nan,
    show: bool = False,
) -> None:
    """
    Heatmap of the best-lag between sensors, in minutes if dt_s is known.

    Entry (i, j) is the lag at which row-sensor i best matches column-sensor j;
    positive means i leads j. The matrix is antisymmetric by construction.
    """
    n = len(columns)
    idx = {c: i for i, c in enumerate(columns)}
    lag = np.full((n, n), np.nan)

    scale = dt_s / 60.0 if np.isfinite(dt_s) else 1.0
    unit = "min" if np.isfinite(dt_s) else "samples"

    for _, r in long_df.iterrows():
        i, j = idx[r["sensor_1"]], idx[r["sensor_2"]]
        lag[i, j] = r["best_lag_samples"] * scale
        lag[j, i] = -r["best_lag_samples"] * scale

    np.fill_diagonal(lag, 0.0)

    vmax = np.nanmax(np.abs(lag)) if np.isfinite(lag).any() else 1.0

    plt.figure(figsize=(12, 10))
    plt.imshow(
        lag,
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
        cmap="coolwarm",
        interpolation="nearest",
    )
    plt.colorbar(label=f"Best lag [{unit}] (row leads column if positive)")
    plt.title("Cross-correlation best-lag matrix")
    plt.xticks(range(n), columns, rotation=90, fontsize=6)
    plt.yticks(range(n), columns, fontsize=6)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()

    plt.close()


def plot_ccf_reference_grid(
    x: pd.DataFrame,
    ref_col: str,
    max_lag: int,
    *,
    save_path: str | Path | None = None,
    dt_s: float = np.nan,
    difference: bool = False,
    ncols: int = 3,
    show: bool = False,
) -> None:
    """
    Grid of cross-correlation curves of every signal against one reference
    column (e.g. generator power). A dashed vertical line marks the lag of the
    peak; with the sign convention here, a peak at positive lag means the
    reference leads that signal.
    """
    if ref_col not in x.columns:
        print(f"[SKIP] reference column '{ref_col}' not present.")
        return

    signals = [c for c in x.columns if c != ref_col]
    if not signals:
        print("[SKIP] no signals to plot against the reference.")
        return

    n = len(signals)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 5.0, nrows * 3.2),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    if np.isfinite(dt_s):
        x_scale = dt_s / 60.0
        x_label = "Lag [min] (ref leads if positive)"
    else:
        x_scale = 1.0
        x_label = "Lag [samples] (ref leads if positive)"

    for index, sig in enumerate(signals):
        ax = axes_flat[index]
        ccf = ccf_pair(x[ref_col], x[sig], max_lag, difference=difference)
        lags = ccf.index.to_numpy() * x_scale
        _, best_lag = _peak(ccf)

        ax.axhline(0.0, color="0.6", linewidth=0.8)
        ax.axvline(0.0, color="0.6", linewidth=0.8)
        ax.plot(lags, ccf.to_numpy(), linewidth=1.2)
        ax.axvline(
            best_lag * x_scale,
            color="0.4",
            linewidth=0.8,
            linestyle="--",
        )
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(sig, fontsize=8)
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"CCF vs {ref_col}", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    for index in range(n, len(axes_flat)):
        axes_flat[index].axis("off")

    kind = "differenced" if difference else "original"
    fig.suptitle(
        f"Cross-correlation against {ref_col} ({kind})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def run_ccf(
    x: pd.DataFrame,
    out_dir: str | Path,
    tag: str,
    *,
    dt_s: float = np.nan,
    max_lag: int = 60,
    difference: bool = False,
    ref_col: str | None = "POWER_kW",
    strong: float = 0.85,
    lead_lag_samples: int = 1,
) -> pd.DataFrame:
    """
    Compute the all-pairs cross-correlation, save the pair tables and a
    best-lag heatmap, and (if a reference column is given) a reference grid.

    'strong' filters pairs by peak correlation magnitude; a pair is flagged as
    a genuine lead-lag relationship when its best lag exceeds lead_lag_samples.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kind = "differenced" if difference else "original"

    long_df = ccf_all_pairs_long(
        x,
        max_lag=max_lag,
        dt_s=dt_s,
        difference=difference,
    )

    strong_df = long_df[
        long_df["peak_corr"].abs() >= strong
    ].copy()

    lead_lag_df = strong_df[
        strong_df["best_lag_samples"].abs() >= lead_lag_samples
    ].copy()

    long_df.to_csv(
        out_dir / f"crosscorr_pairs_{kind}_{tag}.csv",
        index=False,
    )
    strong_df.to_csv(
        out_dir / f"crosscorr_strong_pairs_{kind}_{tag}.csv",
        index=False,
    )
    lead_lag_df.to_csv(
        out_dir / f"crosscorr_leadlag_pairs_{kind}_{tag}.csv",
        index=False,
    )

    plot_lag_heatmap(
        long_df,
        columns=x.columns.tolist(),
        save_path=out_dir / f"crosscorr_lag_heatmap_{kind}_{tag}.png",
        dt_s=dt_s,
    )

    if ref_col is not None:
        plot_ccf_reference_grid(
            x,
            ref_col=ref_col,
            max_lag=max_lag,
            save_path=(
                out_dir / f"crosscorr_vs_{ref_col}_{kind}_{tag}.png"
            ),
            dt_s=dt_s,
            difference=difference,
        )

    print(
        f"[{tag} | {kind}] pairs={len(long_df)} "
        f"strong={len(strong_df)} lead_lag={len(lead_lag_df)}"
    )
    return long_df


def loop_folder(
    base_dir: Path,
    out_root: Path,
    ts_cols: list[str],
    *,
    drop_cols: list[str] | None = None,
    time_col: str = "Created",
    max_lag: int = 60,
    difference: bool = False,
    ref_col: str | None = "POWER_kW",
    strong: float = 0.85,
) -> None:

    base_dir = Path(base_dir)
    out_root = Path(out_root)
    drop_cols = drop_cols or []

    for csv_path in sorted(base_dir.glob("*.csv")):
        tag = csv_path.stem
        out_dir = out_root / tag

        df = pd.read_csv(csv_path)
        dt_s = get_sampling_interval(df, time_col=time_col)

        x = (
            df.select_dtypes(include="number")
            .drop(columns=[*ts_cols, *drop_cols], errors="ignore")
            .copy()
        )
        x = x.loc[:, x.nunique(dropna=True) > 1]

        if x.shape[1] < 2:
            print(f"[SKIP] {tag}: not enough numeric columns ({x.shape[1]}).")
            continue

        run_ccf(
            x,
            out_dir,
            tag=tag,
            dt_s=dt_s,
            max_lag=max_lag,
            difference=difference,
            ref_col=ref_col,
            strong=strong,
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
        / "cross_correlation"
    )

    ts_cols = ["Created", "Modified", "Inserted"]

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
    dt_s = get_sampling_interval(df, time_col="Created")

    # Build the numeric analysis frame (same filtering as loop_folder).
    x = (
        df.select_dtypes(include="number")
        .drop(columns=[*ts_cols, *drop_cols], errors="ignore")
        .copy()
    )
    x = x.loc[:, x.nunique(dropna=True) > 1]

    run_ccf(
        x,
        output_dir,
        tag=f"{dataset_name}_AE_PORT",
        dt_s=dt_s,
        max_lag=60,
        difference=False,
        ref_col="POWER_kW",
        strong=0.9,
    )


if __name__ == "__main__":
    main()