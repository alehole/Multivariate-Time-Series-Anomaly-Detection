import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Example data
# --------------------------------------------------
np.random.seed(42)

t = np.arange(0, 100)  # time index

# Measured generator temperature y_t
y = 70 + 5 * np.sin(0.12 * t) + 0.8 * np.random.randn(len(t))

# Predicted generator temperature y_hat_t
y_hat = 70 + 5 * np.sin(0.12 * t + 0.08)

# Residual e_t = y_t - y_hat_t
e = y - y_hat

# Optional anomaly threshold
threshold = 1.5

# --------------------------------------------------
# Plot 1: Measured vs Predicted
# --------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(t, y, label=r"Measured temperature $y_t$", linewidth=2)
plt.plot(t, y_hat, label=r"Predicted temperature $\hat{y}_t$", linewidth=2, linestyle="--")

# Show residual visually at a few sample points
sample_idx = [15, 35, 60, 85]
for i in sample_idx:
    plt.vlines(t[i], y_hat[i], y[i], linestyles="dotted")
    #plt.text(t[i] + 1, (y[i] + y_hat[i]) / 2, r"$e_t$", fontsize=11)

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
plt.plot(t, e, label=r"Residual $e_t = y_t - \hat{y}_t$", linewidth=2)
plt.axhline(0, linestyle="--", linewidth=1)
plt.axhline(threshold, linestyle="--", linewidth=1, label="Threshold")
plt.axhline(-threshold, linestyle="--", linewidth=1)

# Highlight points outside threshold
anomalies = np.abs(e) > threshold
plt.scatter(
    t[anomalies],
    e[anomalies],
    color="red",
    s=50,
    label="Potential anomalies",
    zorder=3
)

plt.xlabel("Time step")
plt.ylabel("Residual [°C]")
plt.title("Prediction Residual")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()