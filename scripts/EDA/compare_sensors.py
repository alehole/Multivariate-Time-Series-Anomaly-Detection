from pathlib import Path
import config as cfg
import pandas as pd
# ---------------------------------------------------------
# Normalize sensor names
# ---------------------------------------------------------
def normalize_sensor_name(sensor: str) -> str:
    """
    Remove generator-side prefixes so equivalent sensors
    from DS1 and DS2 can be matched.
    """
    sensor = str(sensor).strip()

    prefixes = [
        "AE PORT ",
        "AE STBD ",
        "AE PS ",
        "AE SB ",
        "AE_PS_",
        "AE_SB_",
    ]

    for prefix in prefixes:
        if sensor.startswith(prefix):
            return sensor[len(prefix):]

    return sensor

def main():
    # ---------------------------------------------------------
    # Paths
    # ---------------------------------------------------------


    DS1_PATH = cfg.DATA_PATH / "EDA" / "DS1" / "AE_PORT" / "sensor_statistics_summary_AE_PORT.csv"
    DS2_PATH = cfg.DATA_PATH / "EDA" / "DS2" / "AE_STBD" / "sensor_statistics_summary_AE_STBD.csv"

    OUTPUT_PATH = Path("sensor_mean_comparison.csv")

    # ---------------------------------------------------------
    # Load statistics
    # ---------------------------------------------------------
    ds1 = pd.read_csv(DS1_PATH)
    ds2 = pd.read_csv(DS2_PATH)

    ds1["sensor_key"] = ds1["sensor"].apply(normalize_sensor_name)
    ds2["sensor_key"] = ds2["sensor"].apply(normalize_sensor_name)

    # ---------------------------------------------------------
    # Compare means
    # ---------------------------------------------------------
    comparison = pd.merge(
        ds1[["sensor", "sensor_key", "mean"]],
        ds2[["sensor", "sensor_key", "mean"]],
        on="sensor_key",
        how="inner",
        suffixes=("_DS1", "_DS2"),
    )

    comparison["delta_mean"] = (
            comparison["mean_DS1"]
            - comparison["mean_DS2"]
    )

    comparison["abs_delta_mean"] = comparison["delta_mean"].abs()

    # Optional relative difference
    comparison["delta_mean_percent"] = (
            comparison["delta_mean"]
            / comparison["mean_DS2"].abs()
            * 100.0
    )

    # ---------------------------------------------------------
    # Sort by largest difference
    # ---------------------------------------------------------
    comparison = comparison.sort_values(
        "abs_delta_mean",
        ascending=False,
    )

    # ---------------------------------------------------------
    # Save / print
    # ---------------------------------------------------------
    comparison.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        comparison[
            [
                "sensor_key",
                "mean_DS1",
                "mean_DS2",
                "delta_mean",
                "abs_delta_mean",
                "delta_mean_percent",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved comparison to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()