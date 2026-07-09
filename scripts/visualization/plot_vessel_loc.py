from pathlib import Path
import folium
import numpy as np
import pandas as pd
import config.config as cfg

def get_coordinates(
    df: pd.DataFrame,
    ts_col: str,
    lat_col: str = "cLAT",
    lon_col: str = "cLON",
) -> pd.DataFrame:

    required_cols = [lat_col, lon_col, ts_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    coords = df[[lat_col, lon_col, ts_col]].dropna().copy() # Selects rows with valid coordinates and timestamps
    coords[ts_col] = pd.to_datetime(coords[ts_col], errors="coerce", utc=True) # Converts timestamps to datetime
    coords = coords.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True) # Removes rows with NaN timestamp, sorts by timestamp, resets index
    return coords

def plot_vessel_loc(
    df: pd.DataFrame,
    ts_col: str = "Created",
    lat_col: str = "cLAT",
    lon_col: str = "cLON",
    out_html: str | Path = "vessel_track.html",
    n_mid: int = 100,
) -> None:
    coords = get_coordinates(df, ts_col, lat_col, lon_col)

    if coords.empty:
        print("No valid rows after filtering (lat/lon/timestamp).")
        return

    m = folium.Map(
        location=[coords[lat_col].mean(), coords[lon_col].mean()],
        zoom_start=6,
        #tiles="OpenStreetMap"
        #tiles="CartoDB positron"
        tiles="CartoDB Voyager",
    )

    # Draw track line (usually the most important part)
    folium.PolyLine(
        coords[[lat_col, lon_col]].values,
        color="blue",
        weight=2
    ).add_to(m)

    # --- Choose sparse marker indices: first, evenly spaced middles, last ---
    n = len(coords)
    if n == 1:
        marker_idx = [0]
    else:
        # number of markers total = 2 + n_mid (if possible)
        k = min(n_mid + 2, n)  # cap at n
        marker_idx = np.linspace(0, n - 1, k).astype(int)
        marker_idx = sorted(set(marker_idx.tolist()))  # unique & sorted

    # Add sparse markers
    for i in marker_idx:
        row = coords.iloc[i]
        # Label first/last/middle
        label = "START" if i == 0 else ("END" if i == n - 1 else f"#{i}")
        tooltip = f"{label} | {row[ts_col].isoformat()}"

        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=4 if i in (0, n - 1) else 3,
            color="red" if i in (0, n - 1) else "blue",
            fill=True,
            fill_opacity=0.9,
            tooltip=tooltip
        ).add_to(m)
    Path(out_html).parent.mkdir(parents=True, exist_ok=True)
    m.save(out_html)

def main():
    df_raw = pd.read_csv(cfg.DS1_RAW)
    plot_vessel_loc(df_raw, out_html= "DS1_vessel_track.html")

    df_raw = pd.read_csv(cfg.DS2_RAW)
    plot_vessel_loc(df_raw, out_html= "DS2_vessel_track.html")

if __name__ == "__main__":
    main()
