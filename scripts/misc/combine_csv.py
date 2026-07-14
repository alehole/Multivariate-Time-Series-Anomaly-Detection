from pathlib import Path
import pandas as pd
import config as cfg
from config import DATA_PATH


def create_combined_subsystem_csv(
    base_csv: str | Path,
    extra_csv: str | Path,
    out_csv: str | Path,
    extra_tags: list[str],
    time_col: str = "Created",
    method: str = "nearest",
    tolerance: str | None = "2min",
) -> pd.DataFrame:


    base_csv = Path(base_csv)
    extra_csv = Path(extra_csv)
    out_csv = Path(out_csv)

    base_df = pd.read_csv(base_csv)
    extra_df = pd.read_csv(extra_csv)

    # Check required columns
    if time_col not in base_df.columns:
        raise ValueError(f"{time_col} not found in base CSV")

    if time_col not in extra_df.columns:
        raise ValueError(f"{time_col} not found in extra CSV")

    missing_tags = [tag for tag in extra_tags if tag not in extra_df.columns]
    if missing_tags:
        raise ValueError(
            f"These tags were not found in extra CSV: {missing_tags}\n"
            f"Available LT columns include:\n{list(extra_df.columns)}"
        )

    # Convert timestamps
    base_df[time_col] = pd.to_datetime(base_df[time_col], errors="coerce", utc=True)
    extra_df[time_col] = pd.to_datetime(extra_df[time_col], errors="coerce", utc=True)

    base_df = base_df.dropna(subset=[time_col]).sort_values(time_col)
    extra_df = extra_df.dropna(subset=[time_col]).sort_values(time_col)

    # Keep only timestamp + requested tags from extra subsystem
    extra_df = extra_df[[time_col] + extra_tags]

    # Merge by nearest timestamp
    combined_df = pd.merge_asof(
        base_df,
        extra_df,
        on=time_col,
        direction=method,
        tolerance=pd.Timedelta(tolerance) if tolerance else None,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(out_csv, index=False)

    print(f"Saved combined CSV to: {out_csv}")
    print(f"Shape: {combined_df.shape}")

    return combined_df

def main():
    extra_tags = [
        "LT FW TEMP.",
        "FAN 1 TEMPERATURE",
        "FAN 3 TEMPERATURE",
    ]

    for dataset in (1, 2):
        dataset_name = f"DS{dataset}"

        base_csv = (
            DATA_PATH
            / "subsystems"
            / dataset_name
            / "NUMERIC"
            / "AE_PORT.csv"
        )

        extra_csv = (
            DATA_PATH
            / "raw"
            / dataset_name
            / "LiveData.csv"
        )

        out_csv = (
            DATA_PATH
            / "subsystems"
            / dataset_name
            / "NUMERIC"
            / "AE_PORT_with_extra_TEMP.csv"
        )

        print(f"\nProcessing {dataset_name}...")
        print(f"Base CSV:  {base_csv}")
        print(f"Extra CSV: {extra_csv}")

        create_combined_subsystem_csv(
            base_csv=base_csv,
            extra_csv=extra_csv,
            out_csv=out_csv,
            extra_tags=extra_tags,
        )


if __name__ == "__main__":
    main()

