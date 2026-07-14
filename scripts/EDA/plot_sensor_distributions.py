import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from pathlib import Path
import config as cfg


def scale_data(data: pd.DataFrame, cols, scaler) -> pd.DataFrame:
    scaled_data = data.copy()
    scaled_data[cols] = scaler.fit_transform(data[cols])

    return scaled_data


def plot_per_sensor(
    data,
    cols,
    out_dir,
    prefix="",
    bins=50,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for col in cols:
        plt.figure(figsize=(6, 4))

        sns.histplot(
            data[col].dropna(),
            bins=bins,
            kde=True # Kernel Density Estimate
        )

        mean = data[col].mean()
        std = data[col].std()

        plt.title(f"{col}\nmean={mean:.2f}, std={std:.2f}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.tight_layout()

        safe_name = col.replace(" ", "_").replace(".", "").replace("/", "_")

        plt.savefig(out_dir / f"{prefix}{safe_name}.png", dpi=200)
        plt.close()

    print(f"[OK] Saved {len(cols)} sensor plots to {out_dir}")


def analyze_distribution_folder(
    base_dir: Path,
    out_root: Path,
    bins: int = 50,
):
    for csv_path in sorted(base_dir.glob("*.csv")):
        tag = csv_path.stem

        data = pd.read_csv(csv_path)

        exclude = ["Created", "Modified", "Inserted"]

        cols = (
            data
            .select_dtypes(include="number")
            .drop(columns=exclude, errors="ignore")
            .loc[:, lambda df: df.nunique(dropna=True) > 1]
            .columns
        )

        if len(cols) == 0:
            print(f"[SKIP] {tag}: no numeric non-constant columns.")
            continue

        plot_per_sensor(
            data,
            cols,
            out_dir=out_root / tag / "distributions" / "original",
            prefix="orig_",
            bins=bins,
        )

        robust_scaled = scale_data(data, cols, RobustScaler())
        plot_per_sensor(
            robust_scaled,
            cols,
            out_dir=out_root / tag / "distributions" / "robust_scaled",
            prefix="robust_",
            bins=bins,
        )

        standard_scaled = scale_data(data, cols, StandardScaler())
        plot_per_sensor(
            standard_scaled,
            cols,
            out_dir=out_root / tag / "distributions" / "standard_scaled",
            prefix="standard_",
            bins=bins,
        )

        print(f"[OK] {tag}: distribution plots saved.")


def main():
    DATASET: int = 1

    analyze_distribution_folder(
        base_dir=Path(cfg.DATA_PATH / "subsystems" / f"DS{DATASET}" / "NUMERIC"),
        out_root=Path(cfg.DATA_PATH / "EDA" / f"DS{DATASET}"),
        bins=50,
    )
    DATASET: int = 2
    analyze_distribution_folder(
        base_dir=Path(cfg.DATA_PATH / "subsystems" / f"DS{DATASET}" / "NUMERIC"),
        out_root=Path(cfg.DATA_PATH / "EDA" / f"DS{DATASET}"),
        bins=50,
    )



if __name__ == "__main__":
    main()