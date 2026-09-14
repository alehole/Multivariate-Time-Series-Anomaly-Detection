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
    label_string: str = "with injected anomaly,",
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
        label=label_string,
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

def get_raw_sensor_name(generic_name: str) -> str:
    """Return the DS-specific raw sensor name for a generic sensor name."""

    for raw_name, mapped_name in cfg.RENAME_MAP.items():
        if mapped_name == generic_name:
            return raw_name

    raise KeyError(
        f"Generic sensor '{generic_name}' was not found in cfg.RENAME_MAP."
    )

def main():
    data_path = Path(cfg.DATA_PATH)

    # =====================================================
    # EXPERIMENT CONFIGURATION
    # =====================================================

    # Generic sensor name from config.py
    sensor = "T1"

    # Resolve to DS-specific raw name
    col = get_raw_sensor_name(sensor)

    # Select ONE synthetic fault
    fault = "F4"

    # =====================================================
    # PATHS
    # =====================================================

    input_path = data_path/ "train_test_split"/ f"{cfg.DS}_generator_test.csv"
    output_dir = data_path/ "train_test_split"/"with_anomalies"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir/ f"{cfg.DS}_{fault}_{sensor}_test.csv"
    plot_path = output_dir/ f"{cfg.DS}_{fault}_{sensor}_comparison.png"

    # =====================================================
    # LOAD ORIGINAL TEST DATA
    # =====================================================

    data = pd.read_csv(input_path)
    original_data = data.copy(deep=True)

    # =====================================================
    # INJECT SYNTHETIC ANOMALY
    # =====================================================

    if fault == "F1": # Noise
        start_idx = 100
        end_idx = 1200
        noise_std = 1.0

        data = ai.inject_noise_fault(
            df=data,
            col=col,
            start_idx=start_idx,
            end_idx=end_idx,
            noise_std=noise_std,
        )

        data.loc[start_idx:end_idx - 1, "synthetic_anomaly"] = True
        data.loc[start_idx:end_idx - 1, "synthetic_fault"] = "F1"
        label_string = f"Additive noise ($\\sigma={noise_std}$ °C)"

    elif fault == "F2": # Bias
        start_idx = 200
        bias = 2.0

        data = ai.inject_bias_fault(
            df=data,
            col=col,
            start_idx=start_idx,
            bias=bias,
        )

        data.loc[start_idx:, "synthetic_anomaly"] = True
        data.loc[start_idx:, "synthetic_fault"] = "F2"
        label_string =  f"Constant bias (+{bias:.1f} °C)"

    elif fault == "F3": # Stuck sensor
        start_idx = 1000
        data = ai.inject_stuck_sensor(
            df=data,
            col=col,
            start_idx=start_idx,
        )

        data.loc[start_idx:, "synthetic_anomaly"] = True
        data.loc[start_idx:, "synthetic_fault"] = "F3"
        label_string = "Stuck sensor"

    elif fault == "F4": # Gradual drift
        start_idx = 200
        end_idx = 1000
        final_drift = 3.0

        data = ai.inject_drift_fault(
            df=data,
            col=col,
            start_idx=start_idx,
            end_idx=end_idx,
            final_drift=final_drift,
        )

        data.loc[ start_idx:end_idx - 1, "synthetic_anomaly"] = True
        data.loc[start_idx:end_idx - 1,"synthetic_fault"] = "F5"
        label_string = "Gradual drift ({final_drift:.1f} °C)"


  # =====================================================
    # PLOT
    # =====================================================

    plot_original_vs_modified(
        original_data=original_data,
        modified_data=data,
        col=col,
        time_col=cfg.TS_COL,
        save_path=plot_path,
        show=True,
        label_string=label_string,
    )

    # =====================================================
    # SAVE
    # =====================================================

    data.to_csv(
        output_path,
        index=False,
    )

    print()
    print(f"Dataset : {cfg.DS}")
    print(f"Sensor  : {sensor} -> {col}")
    print(f"Fault   : {fault}")
    print(f"Saved   : {output_path}")


if __name__ == "__main__":
    main()