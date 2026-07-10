import pandas as pd
import numpy as np
from pathlib import Path
import config.config as cfg

def timestamp_report(df, ts_col: str) -> dict:
    s = df[ts_col]      # Extract timestamp column named "ts_col"
    ts = pd.to_datetime(s, format="ISO8601", utc=True, errors="coerce") # Convert s into utc and iso format, NaT if cant be parsed
    ts_valid = ts.dropna() # Removes all invalid timestamps (NaT) from ts.
    dt_s = ts_valid.diff().dropna().dt.total_seconds() # Array storing each sample interval

    total_duration = ts_valid.max() - ts_valid.min()
    total_hours = float(total_duration.total_seconds() / 3600)
    total_days = float(total_duration.total_seconds() / (24 * 3600))

    report = {
        "rows_total": int(len(df)),
        "rows_valid_ts": int(ts_valid.shape[0]),
        "n_invalid_ts": int(ts.isna().sum()),

        "start": str(ts_valid.min()),
        "end": str(ts_valid.max()),
        "total_hours": total_hours,
        "total_days": total_days,

        "avg_sampling_time_s": float(dt_s.mean()),
        "median_sampling_time_s": float(dt_s.median()),
        "p90_sampling_time_s": float(dt_s.quantile(0.90)), # 90% of all sampling intervals are shorter than or equal to this value
        "p95_sampling_time_s": float(dt_s.quantile(0.95)),  # 95% of all sampling intervals are shorter than or equal to this value
        "p99_sampling_time_s": float(dt_s.quantile(0.99)),  # 99% of all sampling intervals are shorter than or equal to this value

        "max_gap_s": float(dt_s.max()),                             # Biggest gap in seconds
        "min_gap_s": float(dt_s.min()),                             # Smallest gap in seconds
        "n_zero_dt": int((dt_s == 0).sum()),

        "n_duplicates": int(ts.duplicated().sum()),
        "n_unique": int(ts.nunique()),
        "is_monotonic_increasing": bool(ts.is_monotonic_increasing),
    }
    return report


def frequency_features(x, dt):

    x = pd.Series(x).dropna().to_numpy()
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    # Remove mean (DC)
    x = x - np.mean(x)

    freqs = np.fft.rfftfreq(len(x), d=dt)
    power = np.abs(np.fft.rfft(x)) ** 2

    # Remove DC component
    freqs = freqs[1:]
    power = power[1:]

    if power.sum() == 0:
        return np.nan, np.nan, np.nan

    # What frequency does this sensor oscillate the most?
    dominant_freq = freqs[np.argmax(power)]

    #How high or low in frequency the energy is concentrated.
    spectral_centroid = np.sum(freqs * power) / np.sum(power)

    # high_freq_energy_ratio: How much of the signal energy lives at high frequencies( noise indicator)
    # High-frequency energy (> 25% of Nyquist)
    hf_mask = freqs > 0.25 * freqs.max()
    hf_energy_ratio = power[hf_mask].sum() / power.sum()

    return dominant_freq, spectral_centroid, hf_energy_ratio

def add_frequency_stats(stats, x, dt):
    freq_rows = {}

    for sensor in stats["sensor"]:
        dom_f, cent_f, hf_ratio = frequency_features(
            x[sensor].values, dt
        )
        freq_rows[sensor] = {
            "dominant_freq_hz": dom_f,
            "spectral_centroid_hz": cent_f,
            "hf_energy_ratio": hf_ratio,
        }

    freq_df = (
        pd.DataFrame.from_dict(freq_rows, orient="index")
        .reset_index()
        .rename(columns={"index": "sensor"})
    )

    return stats.merge(freq_df, on="sensor", how="left")


def sensor_statistics_summary(df):
    X = df.select_dtypes(include="number").copy() # Only numeric columns
    stats = pd.DataFrame(index=X.columns)
    stats["mean"] = X.mean()            # Average value of the sensor over time
    stats["median"] = X.median()        # Median
    stats["std"] = X.std()              # Standard deviation
    stats["min"] = X.min()              # Minimum observed value
    stats["max"] = X.max()              # Maximum observed value
    stats["range"] = stats["max"] - stats["min"]  # Operating range( max − min )

    # 5th and 95th percentiles
    stats["p05"] = X.quantile(0.05)     # 5% of values are below this threshold
    stats["p95"] = X.quantile(0.95)     # 95% of values are below this threshold

    denom = stats["mean"].abs().replace(0, np.nan) # Avoid divide-by-zero in CV
    stats["cv"] = stats["std"] / denom # Coefficient of Variation - how large the variation is compared to the average value.

    # Fraction of unique values:
    # High → continuous / noisy / high-resolution signal
    # Low  → discrete, constant, or slowly changing signal
    stats["unique_ratio"] = X.nunique(dropna=True) / len(X)

    d = X.diff()
    stats["change_rate_mean_abs"] = d.abs().mean()          # avg |Δx|
    stats["zero_change_frac"] = (d == 0).mean()             # fraction of unchanged steps

    stats["missing_frac"] = X.isna().mean() # Fraction of missing samples

    stats["skew"] = X.skew()         # Distribution asymmetry
    stats["kurtosis"] = X.kurtosis() # Tail heaviness / outlier tendency

    return stats.reset_index().rename(columns={"index": "sensor"})


def analyze_folder(
    base_dir: Path,
    out_root: Path,
    ts_col_for_dt: str = "Created",
):
    out_root.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(base_dir.glob("*.csv")):
        tag = csv_path.stem
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(csv_path)

        # Sensor statistics
        sensor_stats = sensor_statistics_summary(df)

        # Frequency stats
        if ts_col_for_dt in df.columns:
            dt_s = timestamp_report(df, ts_col=ts_col_for_dt)["median_sampling_time_s"]
            sensor_stats = add_frequency_stats(sensor_stats, df, dt=dt_s)

        sensor_stats.to_csv(
            out_dir / f"sensor_statistics_summary_{tag}.csv",
            index=False,
        )

        print(f"[OK] {tag}: wrote analysis to {out_dir}")


def main():
    df_raw = pd.read_csv(cfg.DS1_RAW)
    EDA_DIR = cfg.DATA_PATH / "EDA"
    EDA_DIR.mkdir(parents=True, exist_ok=True)

    # --- Timestamp report ---
    for col in ["Created", "Modified", "Inserted"]:
        if col in df_raw.columns:
            ts_report = timestamp_report(df_raw, ts_col=col)
            ts_df = pd.DataFrame([ts_report])

            ts_df.to_csv(EDA_DIR / f"timestamp_report_{col}.csv", index=False)


    analyze_folder(
        base_dir=Path(cfg.DATA_PATH /"subsystems" / "DS1" / "NUMERIC"),
        out_root=Path(EDA_DIR / "DS1"),
        ts_col_for_dt="Created",
    )

    analyze_folder(
        base_dir=Path(cfg.DATA_PATH /"subsystems" / "DS2" / "NUMERIC"),
        out_root=Path(EDA_DIR / "DS2"),
        ts_col_for_dt="Created",
    )



if __name__ == "__main__":
    main()
