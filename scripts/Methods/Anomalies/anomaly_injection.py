import numpy as np
import pandas as pd

def inject_noise_fault(
        df: pd.DataFrame,
        col: str,
        start_idx: int,
        end_idx: int,
        noise_std: float,
        seed: int | None = None,
) -> pd.DataFrame:

    d = df.copy()
    if col not in d.columns:
        raise KeyError(f"{col} not found in dataframe")

    rng = np.random.default_rng(seed)
    end_idx = min(end_idx, len(d) - 1)
    noise = rng.normal(0, noise_std, end_idx - start_idx + 1)

    col_idx = d.columns.get_loc(col)
    d.iloc[start_idx:end_idx + 1, col_idx] = (
        pd.to_numeric(d.iloc[start_idx:end_idx + 1, col_idx], errors="coerce") + noise
    )
    return d


def inject_bias_fault(
    df: pd.DataFrame,
    col: str,
    start_idx: int,
    bias: float,
) -> pd.DataFrame:

    d = df.copy()
    if col not in d.columns:
        raise KeyError(f"{col} not found in dataframe")

    col_idx = d.columns.get_loc(col)
    d.iloc[start_idx:, col_idx] = pd.to_numeric(d.iloc[start_idx:, col_idx], errors="coerce") + bias
    return d


def inject_stuck_sensor(
    df: pd.DataFrame,
    col: str,
    start_idx: int,
) -> pd.DataFrame:

    d = df.copy()
    if col not in d.columns:
        raise KeyError(f"{col} not found in dataframe")

    col_idx = d.columns.get_loc(col)
    value = d.iloc[start_idx, col_idx]
    d.iloc[start_idx:, col_idx] = value
    return d

def inject_drift_fault(
    df: pd.DataFrame,
    col: str,
    start_idx: int,
    end_idx: int,
    final_drift: float,
) -> pd.DataFrame:

    d = df.copy()
    if col not in d.columns:
        raise KeyError(f"{col} not found in dataframe")

    end_idx = min(end_idx, len(d) - 1)

    if start_idx < 0 or start_idx > end_idx:
        raise ValueError("Invalid start_idx/end_idx")

    n = end_idx - start_idx + 1
    drift = np.linspace(0.0, final_drift, n)

    col_idx = d.columns.get_loc(col)

    d.iloc[start_idx:end_idx + 1, col_idx] = (
            pd.to_numeric(d.iloc[start_idx:end_idx + 1, col_idx], errors="coerce").to_numpy()
            + drift
    )
    return d