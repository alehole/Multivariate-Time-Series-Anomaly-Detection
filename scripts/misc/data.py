import pandas as pd

def load_data(csv_file, time_col, sensor_cols, rename_map):
    df = pd.read_csv(csv_file)

    df[time_col] = pd.to_datetime(
        df[time_col],
        format="ISO8601",
        utc=True
    )
    df = df.sort_values(time_col)
    df = df[[time_col] + sensor_cols].dropna()
    df = df.rename(columns=rename_map)
    #df["P"] = df["P"] * 1000.0 # Convert Kwh to W

    return df