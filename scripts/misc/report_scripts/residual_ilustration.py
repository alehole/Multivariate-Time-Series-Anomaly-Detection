import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Example data
# --------------------------------------------------
np.random.seed(42)

t = np.arange(0, 100)
# Measured generator temperature y_t
y = 70 + 5 * np.sin(0.12 * t) + 0.8 * np.random.randn(len(t))
# Predicted generator temperature y_hat_t
y_hat = 70 + 5 * np.sin(0.12 * t + 0.08)

# --------------------------------------------------
# Inject example anomalies
# --------------------------------------------------

# Isolated anomaly -> should NOT be classified as anomaly
y[10] += 3.0
y[20] += 3.0
y[52] += 3.0

# Consecutive positive anomaly -> should be detected
y[45:49] += 3.0

# Consecutive negative anomaly -> should be detected
y[75:79] -= 3.0

# Residual
e = y - y_hat

threshold = 1.5


def consecutive_anomalies(mask, min_consecutive=2):
    confirmed = np.zeros_like(mask, dtype=bool)

    start = 0

    while start < len(mask):
        if mask[start]:
            end = start

            while end < len(mask) and mask[end]:
                end += 1

            if end - start >= min_consecutive:
                confirmed[start:end] = True

            start = end
        else:
            start += 1

    return confirmed


# Individual threshold exceedances
positive_exceeded = e > threshold
negative_exceeded = e < -threshold

threshold_exceeded = positive_exceeded | negative_exceeded

# Require two consecutive exceedances on the SAME side
positive_anomalies = consecutive_anomalies(
    positive_exceeded,
    min_consecutive=2,
)

negative_anomalies = consecutive_anomalies(
    negative_exceeded,
    min_consecutive=2,
)

anomalies = positive_anomalies | negative_anomalies

# --------------------------------------------------
# Plot 1: Measured vs predicted
# --------------------------------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    t,
    y,
    label=r"Measured temperature $y_t$",
    linewidth=2,
)

plt.plot(
    t,
    y_hat,
    label=r"Predicted temperature $\hat{y}_t$",
    linewidth=2,
    linestyle="--",
)

plt.xlabel("Time step")
plt.ylabel("Temperature [°C]")
plt.title("Measured and Predicted Temperature")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# Plot 2: Residual
# --------------------------------------------------
plt.figure(figsize=(10, 4))

plt.plot(
    t,
    e,
    label=r"Residual $e_t = y_t - \hat{y}_t$",
    linewidth=2,
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1,
)

plt.axhline(
    threshold,
    linestyle="--",
    linewidth=1,
    label="Threshold",
)

plt.axhline(
    -threshold,
    linestyle="--",
    linewidth=1,
)

# Optional: show ALL threshold exceedances
plt.scatter(
    t[threshold_exceeded],
    e[threshold_exceeded],
    s=30,
    label="Threshold exceeded",
    zorder=3,
)

# Confirmed anomalies
plt.scatter(
    t[anomalies],
    e[anomalies],
    color="red",
    s=60,
    label="Anomaly",
    zorder=4,
)

plt.xlabel("Time step")
plt.ylabel("Residual [°C]")
plt.title("Prediction Residual")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()