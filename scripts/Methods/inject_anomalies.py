from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import anomaly_injection as ai
import config as cfg


def plot_original_vs_modified(
    original_data: pd.DataFrame,
    modified_data: pd.DataFrame,
    col: str,
    time_col: str = "Created",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot the original and modified sensor signals on the same axes."""

    if time_col not in original_data.columns:
        raise KeyError(f"Time column '{time_col}' was not found.")

    if col not in original_data.columns:
        raise KeyError(f"Sensor column '{col}' was not found.")

    if len(original_data) != len(modified_data):
        raise ValueError(
            "Original and modified dataframes must have the same number of rows."
        )

    time = pd.to_datetime(
        original_data[time_col],
        errors="coerce",
        utc=True,
    )

    original_values = pd.to_numeric(
        original_data[col],
        errors="coerce",
    )

    modified_values = pd.to_numeric(
        modified_data[col],
        errors="coerce",
    )

    valid_mask = (
            time.notna()
            & original_values.notna()
            & modified_values.notna()
    )

    time_plot = time.loc[valid_mask].dt.tz_localize(None)
    original_plot = original_values.loc[valid_mask]
    modified_plot = modified_values.loc[valid_mask]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(
        time_plot,
        original_plot,
        label="Original",
        linewidth=1.5,
        color="tab:blue",
    )

    ax.plot(
        time_plot,
        modified_plot,
        label="With injected anomaly",
        linewidth=1.5,
        color="tab:orange",
        alpha=0.85,
    )

    ax.set_title(f"Original and modified signal: {col}")
    ax.set_xlabel("Time")
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.autofmt_xdate()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    data_path = Path(cfg.DATA_PATH)

    input_path = (
        data_path
        / "train_test_split"
        / "ds1_generator_test.csv"
    )

    output_path = (
        data_path
        / "train_test_split"
        / "with_anomalies"
        / "ds1_generator_test_w_anomalies.csv"
    )

    plot_path = (
        data_path
        / "train_test_split"
        / "with_anomalies"
        / "injected_anomaly_comparison.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------
    # Load dataset
    # --------------------------------
    data = pd.read_csv(input_path)

    # Keep an unchanged copy for comparison.
    original_data = data.copy(deep=True)

    # Sensor to compare in the plot.
    col = "AE PORT GEN.V-WINDING TEMP."

    # --------------------------------
    # Select injected anomalies
    # --------------------------------
    inj_drift_fault = False
    inj_sensor_dropout = False
    inj_noise_fault = False
    inj_bias_fault = True
    inj_stuck_sensor = False

    # --------------------------------
    # Inject anomalies
    # --------------------------------
    if inj_drift_fault:
        data = ai.inject_drift_fault(
            df=data,
            col=col,
            start_idx=200,
            end_idx=1000,
            final_drift=3.0,
        )

    if inj_sensor_dropout:
        data = ai.inject_sensor_dropout(
            df=data,
            col=col,
            start_idx=100,
            end_idx=600,
        )

    if inj_noise_fault:
        data = ai.inject_noise_fault(
            df=data,
            col=col,
            start_idx=100,
            end_idx=600,
            noise_std=1.0,
        )

    if inj_bias_fault:
        data = ai.inject_bias_fault(
            df=data,
            col=col,
            start_idx=200,
            bias=3.0,
        )

    if inj_stuck_sensor:
        data = ai.inject_stuck_sensor(
            df=data,
            col=col,
            start_idx=800,
        )


    # --------------------------------
    # Plot original and modified data
    # --------------------------------
    plot_original_vs_modified(
        original_data=original_data,
        modified_data=data,
        col=col,
        time_col="Created",
        save_path=plot_path,
        show=True,
    )

    # --------------------------------
    # Save modified dataset
    # --------------------------------
    data.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()