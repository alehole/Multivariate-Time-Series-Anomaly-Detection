import numpy as np
import matplotlib.pyplot as plt

def point_outliers():
    # -----------------------------
    # 1. Generate synthetic data
    # -----------------------------
    np.random.seed(42)

    n_points = 200
    # Normal data around mean=10
    data = np.random.normal(loc=10, scale=1.0, size=n_points)

    # Inject a few global/point outliers
    outlier_indices = [50, 170]
    data[outlier_indices] = [20, 18]  # clearly far from the rest

    # -----------------------------
    # 2. Detect global/point outliers (z-score)
    # -----------------------------
    mean = np.mean(data)
    std = np.std(data)

    z_scores = (data - mean) / std

    threshold = 3.0
    is_outlier = np.abs(z_scores) > threshold

    print("Global/point outlier indices:", np.where(is_outlier)[0])
    print("Outlier values:", data[is_outlier])

    # -----------------------------
    # 3. Visualization
    # -----------------------------
    x = np.arange(n_points)

    plt.figure(figsize=(12, 5))
    plt.ylim(7, 22)  # or plt.ylim([ymin, ymax])

    # --- (a) Time-series line plot ---
    plt.plot(x, data, marker='o', linestyle='-', alpha=0.7)
    plt.scatter(x[is_outlier], data[is_outlier], marker='o', s=80, edgecolor='k')
    plt.title("Time Series with Global/Point Outliers")
    plt.xlabel("Time index")
    plt.ylabel("Value")
    # Annotate outliers
    for idx in np.where(is_outlier)[0]:
        plt.annotate("outlier", (idx, data[idx]),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8)
    plt.grid(True)
    plt.show()

    # --- (b) Scatter / value distribution view ---
    plt.figure(figsize=(12, 5))
    plt.ylim(0, 25)  # or plt.ylim([ymin, ymax])
    plt.scatter(x, data, alpha=0.7)
    plt.scatter(x[is_outlier], data[is_outlier], s=80, edgecolor='k')
    plt.axhline(mean, linestyle='--')
    plt.title("Global/Point Outliers in Value Space")
    plt.xlabel("Time index")
    plt.ylabel("Value")

    plt.tight_layout()
    plt.grid(True)
    plt.show()


def contextual_outliers():
    # -----------------------------
    # 1. Generate sine wave + white noise
    # -----------------------------
    np.random.seed(42)

    fs = 1000  # sample rate
    t = np.linspace(0, 2, fs * 2)  # 2 seconds
    freq = 5  # sine frequency

    sine = np.sin(2 * np.pi * freq * t)
    noise = 0.2 * np.random.randn(len(t))
    signal = sine + noise

    # -----------------------------
    # 2. Add contextual anomalies
    # -----------------------------
    signal[250] = -0.35
    signal[550] = 0.5

    # -----------------------------
    # 4. Plot
    # -----------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(t, signal, label="Signal")
    # Plot anomaly circles
    plt.scatter(t[250], signal[250], s=120, facecolor='none',
                edgecolor='red', linewidths=2, label="Contextual anomaly")

    plt.scatter(t[550], signal[550], s=120, facecolor='none',
                edgecolor='red', linewidths=2)


    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Sine Wave with Noise and Contextual Anomalies")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def trend_outliers():
    # -----------------------------
    # 1. Generate sine wave + white noise
    # -----------------------------
    np.random.seed(42)

    fs = 1000  # sample rate
    t = np.linspace(0, 2, fs * 2)  # 2 seconds
    freq = 5  # sine frequency

    sine = np.sin(2 * np.pi * freq * t)
    noise = 0.2 * np.random.randn(len(t))
    signal = sine + noise

    # -----------------------------
    # 2. Add a trend shift anomaly:
    #    ramp up until ~3, then plateau
    # -----------------------------
    shift_start = 800  # index where trend starts
    target_level = 2.0  # approximate plateau amplitude
    ramp_len = 400  # number of samples to ramp up

    trend = np.zeros_like(signal)

    # current level at start of anomaly
    start_level = signal[shift_start]

    # how much we want to lift the signal (approximately)
    delta = target_level - start_level

    # make sure ramp_len does not exceed signal length
    ramp_end = min(shift_start + ramp_len, len(signal))
    actual_ramp_len = ramp_end - shift_start

    # linear ramp from 0 → delta
    ramp = np.linspace(0, delta, actual_ramp_len, endpoint=False)
    trend[shift_start:ramp_end] = ramp

    # plateau: keep the trend constant after the ramp
    if ramp_end < len(signal):
        trend[ramp_end:] = trend[ramp_end - 1]

    # final signal with trend anomaly
    signal_with_trend = signal + trend

    # -----------------------------
    # 3. Plot the result
    # -----------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(t, signal_with_trend, label="Signal")
    plt.axvline(t[shift_start], color='red', linestyle='--', label="Trend shift start")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Trend Outlier")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def shapelet_outliers():
    # -----------------------------
    # 1. Generate sine wave + white noise
    # -----------------------------
    np.random.seed(42)

    fs = 1000  # sample rate
    t = np.linspace(0, 2, fs * 2)  # 2 seconds
    freq = 5  # sine frequency

    sine = np.sin(2 * np.pi * freq * t)
    noise = 0.2 * np.random.randn(len(t))
    signal = sine + noise

    # -----------------------------
    # 2. Insert a SHAPELET anomaly
    #    Replace a local subsequence with a square signal
    # -----------------------------
    start = 800
    end = 900

    # Square shapelet (flat high level)
    square_value = 1.0
    signal[start:end] = square_value

    # -----------------------------
    # 3. Plot the normal signal + shapelet anomaly
    # -----------------------------
    plt.figure(figsize=(12, 5))

    plt.plot(t, signal, label="Signal")

    # Highlight anomaly region
    plt.axvspan(t[start], t[end], color='red', alpha=0.2, label="Shapelet Outlier")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Shapelet Outlier")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def seasonal_outliers():
    # -----------------------------
    # 1. Generate seasonal sine wave + noise
    # -----------------------------
    np.random.seed(42)

    fs = 1000  # sample rate
    duration = 4  # seconds
    t = np.linspace(0, duration, fs * duration)

    freq = 2  # seasonal frequency (2 cycles per second)
    sine = np.sin(2 * np.pi * freq * t)
    noise = 0.2 * np.random.randn(len(t))
    signal = sine + noise

    # -----------------------------
    # 2. Add a SEASONAL outlier
    #    -> One "season" (cycle) breaks the usual pattern
    # -----------------------------
    # Find indices for one specific cycle to distort
    # Let's take the 3rd cycle out of several
    cycle_duration = 1.0 / freq  # seconds per cycle
    outlier_cycle = 3  # which cycle to distort (1-based)

    # Convert cycle window to indices
    start_time = (outlier_cycle - 1) * cycle_duration
    end_time = outlier_cycle * cycle_duration

    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs)

    # Seasonal outlier: suppress amplitude in this cycle
    # (e.g. system did not reach the expected peak)
    seasonal_factor = 0.3  # scale down this cycle strongly
    signal_seasonal_outlier = signal.copy()
    signal_seasonal_outlier[start_idx:end_idx] = (
            sine[start_idx:end_idx] * seasonal_factor
            + noise[start_idx:end_idx]  # still keep noise
    )

    # -----------------------------
    # 3. Plot the seasonal outlier
    # -----------------------------
    plt.figure(figsize=(12, 5))

    plt.plot(t, signal_seasonal_outlier, label="Signal")

    # Highlight the cycle with the seasonal outlier
    plt.axvspan(t[start_idx], t[end_idx], color='red', alpha=0.2,
                label="Seasonal outlier")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Seasonal Outlier")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    point_outliers()
    contextual_outliers()
    trend_outliers()
    shapelet_outliers()
    seasonal_outliers()