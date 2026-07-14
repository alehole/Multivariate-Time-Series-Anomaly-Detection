from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

def total_time_per_state(df, ts_col="Created", state_col="state"):
    d = df.copy()

    # Parse + sort
    d[ts_col] = pd.to_datetime(d[ts_col], errors="coerce", utc=True)
    d = d.sort_values(ts_col).reset_index(drop=True)

    # Duration to next sample (in seconds)
    d["dt_s"] = (d[ts_col].shift(-1) - d[ts_col]).dt.total_seconds()

    # Last row has no "next" timestamp -> drop it (or fill with median)
    d = d.dropna(subset=["dt_s"])

    # Optional: guard against weird negatives / huge gaps
    d = d[(d["dt_s"] >= 0) & (d["dt_s"] < 24*3600)]

    # Sum time per state
    out = d.groupby(state_col)["dt_s"].sum().sort_values(ascending=False)

    # Nice formatting
    return pd.DataFrame({
        "total_seconds": out,
        "total_timedelta": pd.to_timedelta(out, unit="s"),
        "total_hours": out / 3600.0
    })

def classify_harbour_sailing_anchored(
    df: pd.DataFrame,
    sog_col: str = "cSOG",
    rot_col: str = "cROT",
    ts_col: str | None = None,
    *,
    harbour_kn: float = 0.5,          # <= this => "not moving" zone
    sailing_kn: float = 1.0,          # >= this => "moving" zone
    anchor_rot_dpm: float = 5.0,      # deg/min threshold for "swinging"
    persist_minutes: int = 10,
    sample_seconds: int = 60,
    use_abs_rot: bool = True,
    rot_smoothing_minutes: int = 10,
) -> pd.DataFrame:
    """
    States:
      - sailing: SOG >= sailing_kn (after persistence)
      - harbour: low SOG and low ROT (not swinging)
      - anchored: low SOG and high ROT (swinging)
    Notes:
      - cROT is noisy when SOG is very low; we only use it in the low-speed zone.
      - anchor_rot_dpm depends on your data; 3–10 deg/min is common to try.
    """
    d = df.copy()
    if ts_col and ts_col in d.columns:
        d = d.sort_values(ts_col)

    sog = pd.to_numeric(d.get(sog_col), errors="coerce").fillna(0.0)

    rot = pd.to_numeric(d.get(rot_col), errors="coerce").fillna(0.0)
    rot_val = rot.abs() if use_abs_rot else rot
    window_samples = max(
        3,
        int((rot_smoothing_minutes * 60) / sample_seconds),
    )

    rot_val = (
        rot_val
        .rolling(
            window=window_samples,
            center=True,
            min_periods=max(2, window_samples // 2),
        )
        .median()
        .fillna(0.0)
    )

    # --- raw state with hysteresis ---
    raw = []
    prev = "harbour"
    for s, r in zip(sog.to_numpy(), rot_val.to_numpy()):
        if s >= sailing_kn:
            state = "sailing"
        elif s <= harbour_kn:
            # low-speed zone: split harbour vs anchored using ROT
            state = "anchored" if (r >= anchor_rot_dpm) else "harbour"
        else:
            # hysteresis band: keep previous state
            state = prev
        raw.append(state)
        prev = state

    d["state_raw"] = raw

    # --- persistence via rolling majority vote (3-class) ---
    # window in samples
    win = max(3, int((persist_minutes * 60) / sample_seconds))
    minp = max(2, win // 2)

    # encode classes: harbour=0, anchored=1, sailing=2
    code_map = {"harbour": 0, "anchored": 1, "sailing": 2}
    inv_map = {v: k for k, v in code_map.items()}
    d["state_raw_code"] = pd.Series(d["state_raw"]).map(code_map).astype("Int64")

    def _mode(x: pd.Series):
        x = x.dropna()
        if x.empty:
            return np.nan
        # most frequent value; ties -> choose the most recent value in the window
        vc = x.value_counts()
        top = vc[vc == vc.max()].index.to_list()
        if len(top) == 1:
            return int(top[0])
        return int(x.iloc[-1])

    d["state_code"] = d["state_raw_code"].rolling(win, min_periods=minp).apply(_mode, raw=False)
    d["state"] = d["state_code"].map(inv_map)

    # fill initial rows where rolling isn't available yet
    d["state"] = d["state"].fillna(d["state_raw"])

    # helpful debug columns
    d["sog_used"] = sog
    d["rot_used"] = rot_val

    return d

def classify_harbour_sailing(
    df: pd.DataFrame,
    sog_col: str = "cSOG",
    ts_col: str | None = None,
    *,
    harbour_kn: float = 0.5,
    sailing_kn: float = 1.0,
    persist_minutes: int = 10,
    sample_seconds: int = 60,
) -> pd.DataFrame:
    d = df.copy()
    if ts_col and ts_col in d.columns:
        d = d.sort_values(ts_col)

    # --- raw state with hysteresis (string) ---
    raw = []
    prev = "harbour"
    for sog in pd.to_numeric(d[sog_col], errors="coerce").fillna(0.0):
        if sog >= sailing_kn:
            state = "sailing"
        elif sog <= harbour_kn:
            state = "harbour"
        else:
            state = prev
        raw.append(state)
        prev = state
    d["state_raw"] = raw

    # --- numeric encoding for rolling ---
    d["state_raw_bin"] = (d["state_raw"] == "sailing").astype(int)

    # window in samples
    win = max(3, int((persist_minutes * 60) / sample_seconds))
    minp = max(2, win // 2)

    # rolling majority vote: mean > 0.5  => sailing
    roll_mean = d["state_raw_bin"].rolling(win, min_periods=minp).mean()
    d["state_bin"] = (roll_mean > 0.5).astype(int)

    # map back to labels, fall back where rolling not available
    d["state"] = d["state_bin"].map({1: "sailing", 0: "harbour"})
    d["state"] = d["state"].fillna(d["state_raw"])

    return d

def create_sea_state_file(
    df_num: pd.DataFrame,
    output_csv: str | Path,
    plot: bool = True,
) -> pd.DataFrame:
    required_cols = ["Created", "cSOG", "cROT"]
    missing = [c for c in required_cols if c not in df_num.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df_num = df_num.copy()

    # Parse timestamp and remove invalid rows
    df_num["Created"] = pd.to_datetime(
        df_num["Created"],
        errors="coerce",
        utc=True,
    )

    df_num = (
        df_num
        .dropna(subset=["Created"])
        .sort_values("Created")
        .reset_index(drop=True)
    )

    # Classify harbour, sailing, and anchored states
    df = classify_harbour_sailing_anchored(
        df_num,
        sog_col="cSOG",
        rot_col="cROT",
        ts_col="Created",
        harbour_kn=0.8,
        sailing_kn=1.5,
        anchor_rot_dpm=0.2,
        persist_minutes=20,
        sample_seconds=60,
        use_abs_rot=True,
        rot_smoothing_minutes = 2,
    )

    if plot:
        plt.figure(figsize=(12, 4))

        plt.plot(
            df["Created"],
            df["cSOG"],
            label="Speed over ground",
        )

        y_min = df["cSOG"].min()
        y_max = df["cSOG"].max()

        plt.fill_between(
            df["Created"],
            y_min,
            y_max,
            where=df["state"].eq("harbour"),
            alpha=0.2,
            label="Harbour",
        )

        plt.fill_between(
            df["Created"],
            y_min,
            y_max,
            where=df["state"].eq("anchored"),
            alpha=0.2,
            label="Anchored",
        )

        plt.fill_between(
            df["Created"],
            y_min,
            y_max,
            where=df["state"].eq("sailing"),
            alpha=0.2,
            label="Sailing",
        )

        plt.ylabel("SOG [kn]")
        plt.xlabel("Time")
        plt.legend()
        plt.tight_layout()
        plt.show()

    time_cols = [
        c for c in ["Created", "Modified", "Inserted"]
        if c in df.columns
    ]

    out = df[time_cols + ["state"]].copy()

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(f"Saved state classification to: {output_csv}")

    return out


def main():
    ds1_input = (
            Path(cfg.DS1_CATEGORIZED_DIR)
            / "DS1_NUMERIC.csv"
    )

    ds2_input = (
            Path(cfg.DS2_CATEGORIZED_DIR)
            / "DS2_NUMERIC.csv"
    )
    ds1_output = (
        Path(cfg.DATA_PATH)
        / "subsystems"
        / "DS1"
        / "state.csv"
    )

    ds2_output = (
        Path(cfg.DATA_PATH)
        / "subsystems"
        / "DS2"
        / "state.csv"
    )

    # ------------------------------------------------------------
    # DS1
    # ------------------------------------------------------------
    df_ds1 = pd.read_csv(ds1_input)

    state_ds1 = create_sea_state_file(
        df_ds1,
        output_csv=ds1_output,
        plot=True,
    )

    totals_ds1 = total_time_per_state(state_ds1)
    print("\nDS1 total time per state:")
    print(totals_ds1)

    # ------------------------------------------------------------
    # DS2
    # ------------------------------------------------------------
    df_ds2 = pd.read_csv(ds2_input)

    state_ds2 = create_sea_state_file(
        df_ds2,
        output_csv=ds2_output,
        plot=True,
    )

    totals_ds2 = total_time_per_state(state_ds2)
    print("\nDS2 total time per state:")
    print(totals_ds2)

if __name__ == "__main__":
    main()
