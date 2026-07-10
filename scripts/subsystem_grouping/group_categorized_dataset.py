from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import config.config as cfg
from typing import Iterable

def split_csv_by_contains_rules(
    input_csv: str | Path,
    contains_rules: dict[str, str],
    *,
    out_dir: str | Path,
    timestamp_cols: Iterable[str] = ("Created", "Modified", "Inserted"),
    overwrite: bool = True,
) -> dict[str, int]:

    input_csv = Path(input_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Keep only timestamps that actually exist
    ts_cols = [c for c in timestamp_cols if c in df.columns]

    # Prepare rules (case-insensitive)
    rules_upper = {str(k).upper(): str(v) for k, v in contains_rules.items()}
    keys = sorted(rules_upper.keys(), key=len, reverse=True)

    # Group columns
    group_to_cols: dict[str, list[str]] = {}
    unmatched: list[str] = []

    for col in df.columns:
        if col in ts_cols:
            continue

        col_upper = str(col).upper()
        group = None

        for key in keys:
            if key and key in col_upper:
                group = rules_upper[key]
                break

        if group is None:
            unmatched.append(col)
            continue

        group_to_cols.setdefault(group, []).append(col)

    # Helper: safe group name (avoid path traversal / weird separators)
    def safe_name(name: str) -> str:
        return Path(name).name

    written_counts: dict[str, int] = {}

    # Write each group CSV
    for group, cols in sorted(group_to_cols.items()):
        group_clean = safe_name(group)
        out_path = out_dir / f"{group_clean}.csv"

        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Output exists and overwrite=False: {out_path}")

        df_out = df[ts_cols + cols] if ts_cols else df[cols]
        df_out.to_csv(out_path, index=False)
        written_counts[group_clean] = len(cols)

    # Write misc CSV (unmatched)
    misc_clean = safe_name("misc")
    misc_path = out_dir / f"{misc_clean}.csv"
    df_misc = df[ts_cols + unmatched] if ts_cols else df[unmatched]
    df_misc.to_csv(misc_path, index=False)
    written_counts[misc_clean] = len(unmatched)

    # Optional: write unmatched report
    if unmatched:
        report_path = out_dir / "unmatched_columns.csv"
        pd.DataFrame({"column": unmatched}).to_csv(report_path, index=False)
    return written_counts

def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    tag_defs  = load_json(
        Path(cfg.BASE_PROJ_DIR /  "tag_definitions.json")
    )
    timestamp_cols = ("Created", "Modified", "Inserted")

    split_csv_by_contains_rules(
        input_csv=Path(cfg.DS1_CATEGORIZED_DIR / "DS1_NUMERIC.csv") ,
        contains_rules=tag_defs,
        out_dir= Path(cfg.DATA_PATH /"subsystems" / "DS1" / "NUMERIC"),
        timestamp_cols=timestamp_cols,
        overwrite=True,
    )

    split_csv_by_contains_rules(
        input_csv=Path(cfg.DS1_CATEGORIZED_DIR / "DS1_BOOLEAN.csv") ,
        contains_rules=tag_defs,
        out_dir= Path(cfg.DATA_PATH /"subsystems" / "DS1" /"BOOLEAN" ),
        timestamp_cols=timestamp_cols,
        overwrite=True,
    )

    split_csv_by_contains_rules(
        input_csv=Path(cfg.DS2_CATEGORIZED_DIR / "DS2_NUMERIC.csv") ,
        contains_rules=tag_defs,
        out_dir= Path(cfg.DATA_PATH /"subsystems" / "DS2" / "NUMERIC"),
        timestamp_cols=timestamp_cols,
        overwrite=True,
    )

    split_csv_by_contains_rules(
        input_csv=Path(cfg.DS2_CATEGORIZED_DIR / "DS2_BOOLEAN.csv") ,
        contains_rules=tag_defs,
        out_dir= Path(cfg.DATA_PATH /"subsystems" / "DS2" /"BOOLEAN"),
        timestamp_cols=timestamp_cols,
        overwrite=True,
    )

if __name__ == "__main__":
    main()