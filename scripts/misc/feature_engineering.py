import pandas as pd
import numpy as np
import config as cfg

def ts_cols(
    df: pd.DataFrame,
    ts_col: str = "Created",
) -> tuple[pd.DataFrame, float]:
    """
     Clean and analyze timestamp column.

     - converts to datetime
     - removes invalid timestamps
     - removes duplicates
     - sorts data
     - estimates median sampling interval

     Returns
     -------
     cleaned dataframe
     median timestep (seconds)
     """
    if ts_col not in df.columns:
        raise KeyError(f"{ts_col} not in df.columns")

    df = df.copy()

    # Convert to datetime
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)

    print(f"Bad {ts_col} rows:", df[ts_col].isna().sum())

    # Removing Nat+ duplicates + sorting
    df = (
        df.dropna(subset=[ts_col])
        .drop_duplicates(subset=[ts_col])
        .sort_values(ts_col)
        .reset_index(drop=True)
    )

    # Compute timestep
    dt = df[ts_col].diff().dt.total_seconds()

    dt_median = dt.median()
    large_gaps = (dt > 2 * dt_median).sum()
    print("Median dt:", dt_median, "s")
    print("Large gaps:", large_gaps)

    return df , dt_median

# ============================================================
# FEATURE PROCESSING
# https://www.geeksforgeeks.org/machine-learning/feature-transformation-techniques-in-machine-learning/
# https://github.com/xbeat/Machine-Learning/blob/main/Magic%20Squares%20of%20Powers%20in%20Python.md
# ============================================================
def feature_processing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure numeric
    if "POWER_kW" in df.columns:
        power = pd.to_numeric(df["POWER_kW"], errors="coerce")
        df["POWER_kW"] = power
        df["POWER_kW_sq"] = power ** 2
    return df


def calc_power_and_merge(
        df: pd.DataFrame,
        power_counter_name: str,
        output_col="POWER_kW",
) -> pd.DataFrame:

    d = df.copy()

    # Ensure timestamp is datetime & sorted
    d["Created"] = pd.to_datetime(d["Created"], errors="coerce", utc=True)
    d = d.sort_values("Created").reset_index(drop=True)

    kWh_counter_col = power_counter_name
    if kWh_counter_col not in d.columns:
        raise KeyError(f"Missing column: {kWh_counter_col}")

    E_kwh = pd.to_numeric(d[kWh_counter_col], errors="coerce")

    dt_hours = d["Created"].diff().dt.total_seconds() / 3600.0

    # -----------------------------
    # Compute power (kW)= kWh / h
    # -----------------------------
    dE = E_kwh.diff()
    P_kw = dE / dt_hours

    # -----------------------------
    # Clean bad values
    # -----------------------------
    bad = (
            (dt_hours <= 0) |  # invalid time step
            (dE < 0) |  # counter reset / rollover
            (~np.isfinite(P_kw))  # inf / nan
    )
    P_kw[bad] = np.nan

    d[output_col] = P_kw
    return d


def process_generator_file(
    input_csv,
    power_counter_name: str,
) -> None:
    """
    Read, process and overwrite one generator subsystem CSV.
    """
    if not input_csv.exists():
        print(f"[WARNING] File not found: {input_csv}")
        return

    print(f"\nProcessing: {input_csv}")

    df = pd.read_csv(input_csv)

    df = calc_power_and_merge(
        df=df,
        power_counter_name=power_counter_name,
        output_col="POWER_kW",
    )

    df = feature_processing(df)

    df.to_csv(
        input_csv,
        index=False,
    )

    n_valid_power = int(df["POWER_kW"].notna().sum())

    print(f"Saved: {input_csv}")
    print(f"Rows: {len(df):,}")
    print(f"Valid power values: {n_valid_power:,}")


def main():
    generator_files = {
        "AE_PORT.csv": "AE PS POWER COUNTER",
        "AE_STBD.csv": "AE SB POWER COUNTER",
    }

    for dataset in (1, 2):
        dataset_name = f"DS{dataset}"

        subsystem_dir = (
            cfg.DATA_PATH
            / "subsystems"
            / dataset_name
            / "NUMERIC"
        )

        print(f"\n{'=' * 60}")
        print(f"Processing {dataset_name}")
        print(f"{'=' * 60}")

        for filename, counter_name in generator_files.items():
            input_csv = subsystem_dir / filename

            process_generator_file(
                input_csv=input_csv,
                power_counter_name=counter_name,
            )


if __name__ == "__main__":
    main()