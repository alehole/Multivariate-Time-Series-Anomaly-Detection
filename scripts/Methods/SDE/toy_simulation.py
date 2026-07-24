from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
import SDE_config as sde_cfg

from model import f_discrete_implicit


# ---------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------
DT_SECONDS = 60.0
SIMULATION_HOURS = 72.0
MEASUREMENT_NOISE_STD = 0.25
USE_REAL_POWER = True
OUTPUT_DIR = Path(cfg.DATA_PATH)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Synthetic generator power
# ---------------------------------------------------------

if USE_REAL_POWER:
    real_df = pd.read_csv(sde_cfg.train_path)
    real_df["Created"] = pd.to_datetime(real_df["Created"], format="ISO8601", utc=True)
    power = real_df["POWER_kW"].values
    # Elapsed time based on the actual timestamps.
    timestamps = real_df[cfg.TS_COL]
    time_s = (
        timestamps - timestamps.iloc[0]
    ).dt.total_seconds().to_numpy(dtype=float)

    time_h = time_s / 3600.0
    n_samples = len(power)
else:
    time_s = np.arange(
        0.0,
        SIMULATION_HOURS * 3600.0 + DT_SECONDS,
        DT_SECONDS,
    )

    time_h = time_s / 3600.0
    n_samples = len(time_s)

    timestamps = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=n_samples,
        freq=pd.Timedelta(seconds=DT_SECONDS),
        tz="UTC",
    )

    power = np.zeros(n_samples)
    power[(time_h >= 2.0) & (time_h < 10.0)] = 300.0
    power[(time_h >= 10.0) & (time_h < 20.0)] = 500.0
    power[(time_h >= 20.0) & (time_h < 35.0)] = 700.0
    power[(time_h >= 35.0) & (time_h < 45.0)] = 400.0
    power[(time_h >= 45.0) & (time_h < 60.0)] = 650.0
    power[(time_h >= 60.0)] = 250.0

# ---------------------------------------------------------
# Reference temperature
# ---------------------------------------------------------
tref = (20.0+ 2.0 * np.sin( 2.0 * np.pi * time_h / 24.0)
)

dt_steps = np.diff(time_s)
# ---------------------------------------------------------
# True model parameters
# theta = [C1, R1]
# ---------------------------------------------------------
theta_true = np.array([
    120_000.0,  # C1 [kJ/°C]
    0.05,       # R1 [°C/kW]
])


# ---------------------------------------------------------
# Simulate one-state model
# ---------------------------------------------------------
def simulate_generator(
    power_values,
    reference_temperature,
    theta,
    time_steps,
):
    power_values = np.asarray(
        power_values,
        dtype=float,
    )

    reference_temperature = np.asarray(
        reference_temperature,
        dtype=float,
    )

    theta = np.asarray(
        theta,
        dtype=float,
    )

    time_steps = np.asarray(
        time_steps,
        dtype=float,
    )

    if len(power_values) != len(reference_temperature):
        raise ValueError(
            "Power and reference-temperature signals "
            "must have the same length."
        )

    if len(time_steps) != len(power_values) - 1:
        raise ValueError(
            "time_steps must contain one fewer value "
            "than the power signal."
        )

    if np.any(~np.isfinite(power_values)):
        raise ValueError(
            "The power signal contains non-finite values."
        )

    if np.any(~np.isfinite(reference_temperature)):
        raise ValueError(
            "The reference-temperature signal contains "
            "non-finite values."
        )

    states = np.zeros(
        (len(power_values), 1),
        dtype=float,
    )

    _, R1 = theta

    # Initialize at the equilibrium corresponding to
    # the initial power and reference temperature.
    states[0, 0] = (
        reference_temperature[0]
        + R1 * power_values[0]
    )

    for k, dt_k in enumerate(time_steps):
        model_input = np.array([
            power_values[k],
            reference_temperature[k],
        ])

        states[k + 1] = f_discrete_implicit(
            x=states[k],
            u=model_input,
            theta=theta,
            dt=float(dt_k),
        )

    return states

x_true = simulate_generator(
    power_values=power,
    reference_temperature=tref,
    theta=theta_true,
    time_steps=dt_steps,
)
# ---------------------------------------------------------
# Add measurement noise
# ---------------------------------------------------------
rng = np.random.default_rng(cfg.SEED)

temperature_measured = (
    x_true[:, 0]
    + rng.normal(
        loc=0.0,
        scale=MEASUREMENT_NOISE_STD,
        size=len(x_true),
    )
)

# ---------------------------------------------------------
# Create dataframe
# ---------------------------------------------------------


df = pd.DataFrame({
    cfg.TS_COL: timestamps,
    "time_s": time_s,
    "time_h": time_h,
    "P": power,
    "Tref": tref,
    "T1": temperature_measured,
    "T1_true": x_true[:, 0],
})

# ---------------------------------------------------------
# Chronological train/test split
# ---------------------------------------------------------
split_index = int(0.70 * len(df))

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()


# ---------------------------------------------------------
# Save CSV files
# ---------------------------------------------------------
full_path = OUTPUT_DIR / sde_cfg.CSV_FULL
train_path = OUTPUT_DIR / sde_cfg.CSV_TRAIN
test_path = OUTPUT_DIR / sde_cfg.CSV_TEST

df.to_csv(full_path, index=False)
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Saved complete dataset: {full_path}")
print(f"Saved training dataset: {train_path}")
print(f"Saved test dataset:     {test_path}")

print(f"\nTotal observations: {len(df):,}")
print(f"Training observations: {len(train_df):,}")
print(f"Test observations: {len(test_df):,}")

print("\nTrue parameters:")
print(f"C1 = {theta_true[0]:.1f} kJ/°C")
print(f"R1 = {theta_true[1]:.4f} °C/kW")


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
fig, axes = plt.subplots(
    2,
    1,
    figsize=(12, 8),
    sharex=True,
)

axes[0].plot(
    time_h,
    power,
    label="Generator power",
)
axes[0].set_ylabel("Power [kW]")
axes[0].set_title("Synthetic generator load")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(
    time_h,
    temperature_measured,
    label="Synthetic T1 with measurement noise",
)

axes[1].plot(
    time_h,
    x_true[:, 0],
    label="Noise-free synthetic T1",
)
axes[1].plot(
    time_h,
    tref,
    linestyle="--",
    label="Ambient reference temperature",
)

axes[1].set_xlabel("Time [hours]")
axes[1].set_ylabel("Temperature [°C]")
axes[1].set_title("Synthetic temperature response of the one-state RC model")
axes[1].grid(True)
axes[1].legend()

plt.tight_layout()
plt.show()