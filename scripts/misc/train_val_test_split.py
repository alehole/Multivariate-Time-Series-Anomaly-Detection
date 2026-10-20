from pathlib import Path
import pandas as pd
import config as cfg

def split_csv_train_val_test(
    input_csv: str | Path,
    train_csv: str | Path,
    val_csv: str | Path,
    test_csv: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/validation/test split by row position.

    The test ratio is the remainder (1 - train_ratio - val_ratio).
    Assumes the CSV is already ordered by acquisition time.
    """
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    df = pd.read_csv(input_csv)
    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    for path, part in (
        (train_csv, train_df),
        (val_csv, val_df),
        (test_csv, test_df),
    ):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(path, index=False)

    test_ratio = 1 - train_ratio - val_ratio
    print(f"Total rows : {n:,}")
    print(f"Train rows : {len(train_df):,} ({len(train_df)/n:.1%})")
    print(f"Val rows   : {len(val_df):,} ({len(val_df)/n:.1%})")
    print(f"Test rows  : {len(test_df):,} ({len(test_df)/n:.1%})")

    return train_df, val_df, test_df


def main():
    output_dir = cfg.DATA_PATH / "train_test_split"
    output_dir.mkdir(parents=True, exist_ok=True)
    # -----------------------------------------------------
    # DS1
    # -----------------------------------------------------
    ds1_generator_path = cfg.DATA_PATH / "subsystems" / f"DS{1}" / "NUMERIC" / "AE_PORT.csv"
    ds1_train_generator_path_output = cfg.DATA_PATH / "train_test_split" / f"ds{1}_generator_train.csv"
    ds1_val_generator_path_output = cfg.DATA_PATH / "train_test_split" / f"ds{1}_generator_val.csv"
    ds1_test_generator_path_output = cfg.DATA_PATH / "train_test_split" / f"ds{1}_generator_test.csv"

    split_csv_train_val_test(
        input_csv=ds1_generator_path,
        train_csv=ds1_train_generator_path_output,
        val_csv=ds1_val_generator_path_output,
        test_csv=ds1_test_generator_path_output,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    # -----------------------------------------------------
    # DS2
    # -----------------------------------------------------
    ds2_generator_path = cfg.DATA_PATH / "subsystems" / f"DS{2}" / "NUMERIC" / "AE_STBD.csv"
    ds2_train_generator_path_output = cfg.DATA_PATH / "train_test_split" / f"ds{2}_generator_train.csv"
    ds2_val_generator_path_output = cfg.DATA_PATH / "train_test_split" / f"ds{2}_generator_val.csv"
    ds2_test_generator_path_output = cfg.DATA_PATH / "train_test_split" / f"ds{2}_generator_test.csv"

    split_csv_train_val_test(
        input_csv=ds2_generator_path,
        train_csv=ds2_train_generator_path_output,
        val_csv=ds2_val_generator_path_output,
        test_csv=ds2_test_generator_path_output,
        train_ratio=0.7,
        val_ratio=0.15,
    )

if __name__ == '__main__':
    main()