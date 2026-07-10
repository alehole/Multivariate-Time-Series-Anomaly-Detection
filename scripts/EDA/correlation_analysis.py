import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.config import paths


def plot_corr(corr: pd.DataFrame,
              *,
              save_path: str | Path | None = None,
              show: bool = False
              ):

    plt.figure(figsize=(12, 10))
    plt.imshow(
        corr.values,
        aspect="auto",
        vmin=-1, vmax=1,
        cmap="coolwarm",
        interpolation="nearest",
    )
    plt.colorbar(label="Correlation coefficient")
    plt.title("Correlation Matrix")

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=6)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()

    plt.close()


def corr_to_long(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Upper triangle (no diagonal) -> long table sensor_1, sensor_2, correlation."""
    mask = np.triu(np.ones(corr_df.shape), k=1).astype(bool)
    return (
        corr_df.where(mask)
        .stack()
        .reset_index()
        .rename(columns={"level_0": "sensor_1", "level_1": "sensor_2", 0: "correlation"})
    )


def run_corr(
    x: pd.DataFrame,
    out_dir: str | Path,
    tag: str,
    method: str = "pearson",
    strong: float = 0.85,
    extremely_strong: float = 0.98,
) -> pd.DataFrame:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corr = x.select_dtypes(include="number").corr(method=method)

    long_df = corr_to_long(corr)
    strong_df = long_df[long_df["correlation"].abs() >= strong].copy()
    extremely_strong_df = strong_df[strong_df["correlation"].abs() >= extremely_strong].copy()

    pairs_a = strong_df[["sensor_1", "correlation"]].rename(columns={"sensor_1": "sensor"})
    pairs_b = strong_df[["sensor_2", "correlation"]].rename(columns={"sensor_2": "sensor"})
    pairs_all = pd.concat([pairs_a, pairs_b], ignore_index=True)

    corr_summary = (
        pairs_all.assign(abs_corr=lambda d: d["correlation"].abs())
        .groupby("sensor")
        .agg(
            n_strong_corr=("abs_corr", "count"),
            max_corr=("abs_corr", "max"),
            min_corr=("abs_corr", "min"),
            std_corr=("abs_corr", "std"),
            mean_corr=("abs_corr", "mean"),
            median_corr=("abs_corr", "median"),
        )
        .reset_index()
        .sort_values("n_strong_corr", ascending=False)
    )

    strong_df.to_csv(out_dir / f"correlation_strong_pairs_{tag}.csv", index=False)
    extremely_strong_df.to_csv(out_dir / f"correlation_extremely_strong_pairs_{tag}.csv", index=False)
    corr_summary.to_csv(out_dir / f"correlation_sensor_summary_{tag}.csv", index=False)
    corr.to_csv(out_dir / f"correlation_matrix_{tag}_full.csv")

    print(
        f"[{tag} | {method}] sensors={corr.shape[0]} "
        f"strong_pairs={len(strong_df)} extremely_strong_pairs={len(extremely_strong_df)}"
    )
    return corr


def loop_folder(
    base_dir: Path,
    out_root: Path,
    ts_cols: list[str],
    show_plot: bool = False,
    method: str = "pearson",
    strong: float = 0.85,
    extremely_strong: float = 0.98,
):

    for csv_path in sorted(base_dir.glob("*.csv")):
        tag = csv_path.stem
        out_dir = out_root / tag

        df = pd.read_csv(csv_path)

        X = (
            df.select_dtypes(include="number")
            .drop(columns=ts_cols, errors="ignore")
            .copy()
        )

        # remove constant columns
        #X = X.loc[:, X.nunique(dropna=True) > 1]

        if X.shape[1] < 2:
            print(f"[SKIP] {tag}: not enough numeric columns to correlate ({X.shape[1]}).")
            continue

        corr_df = run_corr(
            X,
            out_dir,
            tag=tag,
            method=method,
            strong=strong,
            extremely_strong=extremely_strong,
        )

        plot_corr(
            corr_df,
            save_path=out_dir / f"correlation_matrix_{tag}.png",
            show=show_plot,
        )

def main():
    if False:
        base_dir = paths.split_NUMERIC_DIR
        out_path = paths.EDA_DIR

        ts = ["Created", "Modified", "Inserted"]

        loop_folder(
            base_dir,
            out_path,
            ts,
            show_plot=False,
            method="spearman",
            strong=0.9,
            extremely_strong=0.98,
        )

        #path = "../Correlation/AE_PORT/corr_matrix_AE_PORT_full.csv"
        #df = pd.read_csv(path, index_col=0)
        #plot_corr(df, show=True)

    if True:
        base_dir = paths.split_temp_DIR
        out_path = paths.EDA_DIR

        ts = ["Created", "Modified", "Inserted"]

        loop_folder(
            base_dir,
            out_path,
            ts,
            show_plot=False,
            method="spearman",
            strong=0.9,
            extremely_strong=0.98,
        )



if __name__ == "__main__":
    main()
