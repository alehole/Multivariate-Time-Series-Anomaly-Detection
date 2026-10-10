import numpy as np
import pandas as pd
import torch
from torch import Tensor
from sklearn.preprocessing import RobustScaler

def create_profiles(
    df: pd.DataFrame,
    ts_col: str = "Created",
    *,
    window_steps: int = 300,
    dt_s: float = 60.0,
    gap_factor: float = 3.0,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Divide dataframe into fixed-length sequence profiles.

    Profiles are also split whenever a timestamp gap exceeds
    gap_factor * dt_s.
    """

    df = df.copy()

    if ts_col not in df.columns:
        raise KeyError(f"{ts_col} not found in dataframe.")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    # -----------------------------------------------------
    # Detect discontinuities
    # -----------------------------------------------------
    delta_t = df[ts_col].diff().dt.total_seconds()
    gap_threshold = gap_factor * dt_s
    new_segment = delta_t.isna() | (delta_t > gap_threshold)

    df["_segment_id"] = new_segment.cumsum() - 1

    # -----------------------------------------------------
    # Make fixed-length profiles within each segment
    # -----------------------------------------------------
    profile_ids = np.empty(len(df), dtype=np.int64)
    next_profile_id = 0

    for _, segment in df.groupby("_segment_id", sort=False):

        indices = segment.index.to_numpy()

        for start in range(0, len(indices), window_steps):
            idx = indices[start:start + window_steps]

            profile_ids[idx] = next_profile_id
            next_profile_id += 1

    df["profile_id"] = profile_ids
    df = df.drop(columns="_segment_id")

    profiles = sorted(df["profile_id"].unique())

    return df, profiles

def train_val_test_split_profiles(
    df: pd.DataFrame,
    ts_col: str = "Created",
    *,
    window_steps: int = 300,
    val_profile_len: int = 3,
    test_profile_len: int = 3,
    dt_s: float = 60,
) -> tuple[pd.DataFrame, list[int], list[int], list[int], pd.Series]:

    df = df.copy()
    if ts_col not in df.columns:
        raise KeyError(f"{ts_col} not in df.columns")


    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)

    t0 = df[ts_col].min()
    window_s = window_steps * dt_s
    df["profile_id"] = ((df[ts_col] - t0).dt.total_seconds() // window_s).astype("int64")

    all_profiles = sorted(df["profile_id"].unique())
    if len(all_profiles) <= val_profile_len + test_profile_len:
        raise ValueError("Not enough profiles for train/val/test split")

    test_profiles = all_profiles[-test_profile_len:]
    val_profiles = all_profiles[-(val_profile_len + test_profile_len):-test_profile_len]
    train_profiles = all_profiles[:-(val_profile_len + test_profile_len)]

    profile_sizes = df.groupby("profile_id").size()
    print(f"Total profiles: {len(all_profiles)}")
    print(f"Train profiles: {len(train_profiles)}")
    print(f"Val profiles:   {len(val_profiles)}")
    print(f"Test profiles:  {len(test_profiles)}")

    return df, train_profiles, val_profiles, test_profiles, profile_sizes

from sklearn.preprocessing import RobustScaler

def scale_train_test_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    input_cols: list[str],
    target_cols: list[str],
):
    """
    Fit scalers only on the training dataset and apply them
    to both the training and test datasets.
    """
    train_data = train_data.copy()
    test_data = test_data.copy()

    all_cols = input_cols + target_cols

    for df in (train_data, test_data):
        df[all_cols] = df[all_cols].apply(
            pd.to_numeric,
            errors="coerce",
        )

        df[input_cols] = (
            df.groupby("profile_id")[input_cols]
            .transform(lambda group: group.ffill().bfill())
        )

        df[target_cols] = (
            df.groupby("profile_id")[target_cols]
            .transform(lambda group: group.ffill().bfill())
        )

    x_scaler = RobustScaler()
    y_scaler = RobustScaler()

    # Fit only on training data.
    x_scaler.fit(train_data[input_cols])
    y_scaler.fit(train_data[target_cols])

    train_data[input_cols] = x_scaler.transform(
        train_data[input_cols]
    )
    train_data[target_cols] = y_scaler.transform(
        train_data[target_cols]
    )

    test_data[input_cols] = x_scaler.transform(
        test_data[input_cols]
    )
    test_data[target_cols] = y_scaler.transform(
        test_data[target_cols]
    )

    return train_data, test_data, x_scaler, y_scaler
def scale_profile_data(
    df: pd.DataFrame,
    train_profiles: list[int],
    input_cols: list[str],
    target_cols: list[str],
) -> tuple[pd.DataFrame, RobustScaler, RobustScaler]:

    df = df.copy()
    train_mask = df["profile_id"].isin(train_profiles)

    # Ensure numeric
    df[input_cols] = df[input_cols].apply(pd.to_numeric, errors="coerce")
    df[target_cols] = df[target_cols].apply(pd.to_numeric, errors="coerce")

    df[input_cols] = df.groupby("profile_id")[input_cols].ffill().bfill()
    df[target_cols] = df.groupby("profile_id")[target_cols].ffill().bfill()

    # Initialize scalers
    x_scaler = RobustScaler()
    y_scaler = RobustScaler()

    # Fit on training data only
    x_scaler.fit(df.loc[train_mask, input_cols])
    y_scaler.fit(df.loc[train_mask, target_cols])

    # Transform entire dataset
    df[input_cols] = x_scaler.transform(df[input_cols])
    df[target_cols] = y_scaler.transform(df[target_cols])

    print(f"Scaling {len(input_cols)} inputs and {len(target_cols)} targets")

    return df, x_scaler, y_scaler


def tensorize_profiles(
    df: pd.DataFrame,
    profiles: list[int],
    input_cols: list[str],
    target_cols: list[str],
    device: torch.device | str = "cpu",
) -> tuple[Tensor, Tensor, Tensor]:

    cols = input_cols + ["profile_id"] + target_cols
    data = df[cols].copy()

    profile_sizes = data.groupby("profile_id").size()
    max_T = int(profile_sizes.loc[profiles].max())

    B = len(profiles)
    F = len(input_cols) + len(target_cols)

    tensor = np.full((B, max_T, F), np.nan, dtype=np.float32)

    grouped = data[data["profile_id"].isin(profiles)].groupby("profile_id", sort=False)

    for b, pid in enumerate(profiles):
        g = grouped.get_group(pid).sort_index()
        arr = (
            g.drop(columns="profile_id")
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=np.float32)
        )
        tensor[b, :len(arr), :] = arr

    mask = ~np.all(np.isnan(tensor), axis=2)
    tensor = np.nan_to_num(tensor, nan=0.0).astype(np.float32)

    x = tensor[:, :, :len(input_cols)]
    y = tensor[:, :, len(input_cols):]

    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
        torch.from_numpy(mask).to(device),
    )



def scale_train_val_test_data(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    input_cols: list[str],
    target_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler, RobustScaler]:
    """
    Fit scalers on the training split only and apply them to all three splits.
    Each split must already contain a 'profile_id' column (from create_profiles).
    """
    train_data = train_data.copy()
    val_data = val_data.copy()
    test_data = test_data.copy()

    all_cols = input_cols + target_cols

    for df in (train_data, val_data, test_data):
        df[all_cols] = df[all_cols].apply(pd.to_numeric, errors="coerce")
        df[input_cols] = (
            df.groupby("profile_id")[input_cols]
            .transform(lambda g: g.ffill().bfill())
        )
        df[target_cols] = (
            df.groupby("profile_id")[target_cols]
            .transform(lambda g: g.ffill().bfill())
        )

    x_scaler = RobustScaler()
    y_scaler = RobustScaler()

    # Fit on training data only.
    x_scaler.fit(train_data[input_cols])
    y_scaler.fit(train_data[target_cols])

    for df in (train_data, val_data, test_data):
        df[input_cols] = x_scaler.transform(df[input_cols])
        df[target_cols] = y_scaler.transform(df[target_cols])

    return train_data, val_data, test_data, x_scaler, y_scaler