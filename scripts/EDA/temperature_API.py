import pandas as pd
import requests
from matplotlib.path import Path
from pathlib import Path
import config as cfg

def fetch_weather_hourly(
        latitude: float,
        longitude: float,
        start_date: str,
        end_date: str
) -> pd.DataFrame:

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "timezone": "UTC"
    }
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    if not times or not temps:
        raise ValueError("No weather data returned from Open-Meteo.")
    return pd.DataFrame(
        {
            "weather_hour": pd.to_datetime(times, utc=True, errors="coerce"),
            "ambient_temp": temps,
        }
    ).dropna(subset=["weather_hour"])


def main():
    required_cols = ["Created", "cLAT", "cLON"]

    for dataset in (1, 2):
        dataset_name = f"DS{dataset}"

        csv_path = (
            Path(cfg.DATA_PATH)
            / "raw_categorized"
            / dataset_name
            / f"{dataset_name}_NUMERIC.csv"
        )

        output_csv = (
            Path(cfg.DATA_PATH)
            / "subsystems"
            / dataset_name
            / "ambient_temperature.csv"
        )

        print(f"\nProcessing {dataset_name}...")
        print(f"Input:  {csv_path}")
        print(f"Output: {output_csv}")

        scada_df = pd.read_csv(csv_path)

        missing = [
            column
            for column in required_cols
            if column not in scada_df.columns
        ]

        if missing:
            raise KeyError(
                f"{dataset_name} is missing required columns: {missing}"
            )

        scada_df["Created"] = pd.to_datetime(
            scada_df["Created"],
            utc=True,
            errors="coerce",
        )

        scada_df = (
            scada_df
            .dropna(subset=required_cols)
            .sort_values("Created")
        )

        if scada_df.empty:
            raise ValueError(
                f"{dataset_name} contains no valid timestamp or "
                "coordinate rows."
            )

        lat = float(scada_df["cLAT"].mean())
        lon = float(scada_df["cLON"].mean())

        date_start = scada_df["Created"].min().date().isoformat()
        date_end = scada_df["Created"].max().date().isoformat()

        weather_df = fetch_weather_hourly(
            lat,
            lon,
            date_start,
            date_end,
        )

        weather_df["weather_hour"] = pd.to_datetime(
            weather_df["weather_hour"],
            utc=True,
            errors="coerce",
        )

        weather_df = (
            weather_df
            .dropna(subset=["weather_hour"])
            .sort_values("weather_hour")
        )

        merged = pd.merge_asof(
            scada_df,
            weather_df,
            left_on="Created",
            right_on="weather_hour",
            direction="nearest",
            tolerance=pd.Timedelta("30min"),
        )

        output_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        merged[
            ["Created", "cLAT", "cLON", "ambient_temp"]
        ].to_csv(
            output_csv,
            index=False,
        )

        matched = merged["ambient_temp"].notna().sum()

        print(
            f"Saved {len(merged):,} rows for {dataset_name}. "
            f"Weather matches: {matched:,}/{len(merged):,}"
        )


if __name__ == "__main__":
    main()