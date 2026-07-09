
import pandas as pd
from pathlib import Path
import config.config as cfg
def constant_datapoints(df):
    constant = df.columns[df.nunique(dropna=False) == 1].tolist()
    non_constant = df.columns[df.nunique(dropna=False) > 1].tolist()

    print("Constant:", len(constant))
    print("Non-constant:", len(non_constant))
    return constant

def timestamp_columns(df):

    timestamp_cols = [
        col for col in df.columns
        if df[col].astype(str).str.contains(
            r"\d{4}-\d{2}-\d{2}[T ]",
            regex=True
        ).any()
    ]

    if not timestamp_cols:
        print("No timestamp columns found.")
        return pd.DataFrame()

    ts_df = df[timestamp_cols].copy()

    for col in timestamp_cols:
        ts_df[col] = pd.to_datetime(
            ts_df[col],
            errors="coerce",
            utc=True,
            format="mixed",
        )

    print("Timestamp columns:", len(timestamp_cols))
    return ts_df

def categorical_columns(
        df,
        timestamp_cols=None
):

    categorical_cols = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if timestamp_cols:
        categorical_cols = [
            c for c in categorical_cols if c not in timestamp_cols
        ]

    if not categorical_cols:
        print("No categorical columns found.")
        return pd.DataFrame()
    print("Categorical (string) columns:", len(categorical_cols))

    # Build selected columns safely
    selected_columns = []
    if timestamp_cols:
        selected_columns += timestamp_cols
    selected_columns += categorical_cols

    cat_df = df[selected_columns].copy()

    return cat_df

def boolean_columns(
        df,
        timestamp_cols=None
):
    boolean_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    if not boolean_cols:
        print("No boolean columns found.")
        return pd.DataFrame()
    print("Boolean columns:", len(boolean_cols))

    # Build selected columns safely
    selected_columns = []
    if timestamp_cols:
        selected_columns += timestamp_cols
    selected_columns += boolean_cols
    bool_df = df[selected_columns].copy()

    # Convert bool → int
    bool_df[boolean_cols] = bool_df[boolean_cols].astype(int)

    return bool_df

def numeric_columns(
        df,
        timestamp_cols=None
):
    df_num = df.copy()

    # Detect numeric columns
    numeric_cols = df_num.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        print("No numeric columns found.")
        return pd.DataFrame()

    print("Numeric columns:", len(numeric_cols))

    # Build selected columns safely
    selected_columns = []
    if timestamp_cols:
        selected_columns += timestamp_cols
    selected_columns += numeric_cols

    num_df = df_num[selected_columns].copy()
    return num_df

def categorize_dataset(raw_csv, out_dir, prefix):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_csv, sep=",")

    # Detect and store constant variables
    constant_cols = constant_datapoints(df)

    pd.DataFrame({"constant_columns": constant_cols}).to_csv(
        out_dir / f"{prefix}_CONSTANT.csv",
        index=False,
    )

    # Drop constant variables
    df = df.drop(columns=constant_cols)

    # Detect and store timestamp columns
    ts_df = timestamp_columns(df)
    ts_df.to_csv(out_dir / f"{prefix}_TIMESTAMPS.csv", index=False)
    timestamp_cols = ts_df.columns.tolist()

    # Detect and store boolean columns
    bool_df = boolean_columns(df, timestamp_cols)
    bool_df.to_csv(out_dir / f"{prefix}_BOOLEAN.csv", index=False)

    # Detect and store numeric columns
    num_df = numeric_columns(df, timestamp_cols)
    num_df.to_csv(out_dir / f"{prefix}_NUMERIC.csv", index=False)

    # Detect and store categorical columns
    cat_df = categorical_columns(df, timestamp_cols)
    cat_df.to_csv(out_dir / f"{prefix}_CATEGORICAL.csv", index=False)


def main():
    categorize_dataset(
        raw_csv=cfg.DS1_RAW,
        out_dir=cfg.DS1_CATEGORIZED_DIR,
        prefix="DS1",
    )
    categorize_dataset(
        raw_csv=cfg.DS2_RAW,
        out_dir=cfg.DS2_CATEGORIZED_DIR,
        prefix="DS2",
    )


if __name__ == "__main__":
    main()