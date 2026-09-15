import numpy as np
import matplotlib.pyplot as plt


def generate_base_signal(n=500):
    t = np.arange(n)
    y = 10 + 2 * np.sin(2 * np.pi * t / 100)
    return t, y


def inject_additive_noise(y, start, end, noise_std=0.8):
    y_fault = y.copy()
    y_fault[start:end] += np.random.normal(0, noise_std, end - start)
    return y_fault


def inject_constant_bias(y, start, bias=2.0):
    y_fault = y.copy()
    y_fault[start:] += bias
    return y_fault


def inject_stuck_sensor(y, start):
    y_fault = y.copy()
    y_fault[start:] = y_fault[start]
    return y_fault


def inject_sensor_dropout(y, start, end):
    y_fault = y.copy()
    y_fault[start:end] = np.nan
    return y_fault


def inject_gradual_drift(y, start, end, final_drift=3.0):
    y_fault = y.copy()
    drift = np.linspace(0, final_drift, end - start)
    y_fault[start:end] += drift
    return y_fault


def plot_fault_example(ax, t, y, y_fault, title, start=None, end=None):
    ax.plot(t, y, label="Original signal", linewidth=2)
    ax.plot(t, y_fault, label="Faulty signal", linewidth=2)

    if start is not None:
        ax.axvline(start, linestyle="--", linewidth=1)
    if end is not None:
        ax.axvline(end, linestyle="--", linewidth=1)

    ax.set_title(title)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Signal value")
    ax.grid(True, alpha=0.3)
    ax.legend()


def main():
    np.random.seed(42)

    t, y = generate_base_signal(n=500)

    start = 150
    end = 300

    y_f1 = inject_additive_noise(y, start, end, noise_std=0.8)
    y_f2 = inject_constant_bias(y, start, bias=2.0)
    y_f3 = inject_stuck_sensor(y, start)
    y_f4 = inject_gradual_drift(y, start, end, final_drift=3.0)

    fig, axes = plt.subplots(4, 1, figsize=(10, 16), sharex=True)

    plot_fault_example(
        axes[0], t, y, y_f1,
        "F1: Additive noise",
        start, end
    )

    plot_fault_example(
        axes[1], t, y, y_f2,
        "F2: Constant bias",
        start, None
    )

    plot_fault_example(
        axes[2], t, y, y_f3,
        "F3: Stuck sensor",
        start, None
    )

    plot_fault_example(
        axes[3], t, y, y_f4,
        "F4: Gradual drift",
        start, end
    )

    fig.suptitle(
        "Illustration of synthetic fault types (F1--F4)",
        fontsize=14
    )

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    plt.show()

if __name__ == "__main__":
    main()