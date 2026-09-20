import numpy as np
import pandas as pd

import config as cfg
import SDE_config as sde_cfg

from thermal_models import f_continuous as evaluate_continuous_model

def f_continuous(
    x,
    u,
    theta,
    model_option=None,
):
    """Evaluate the selected continuous-time thermal model."""

    if model_option is None:
        model_option = sde_cfg.MODEL_OPTION

    return evaluate_continuous_model(
        x=x,
        u=u,
        theta=theta,
        model_option=model_option,
    )


def f_discrete_explicit(x, u, theta, dt):
    """Discretize the model using Forward Euler."""
    x = np.asarray(x, dtype=float).ravel()
    return x + dt * f_continuous(x=x, u=u,theta=theta)

# Discretize the continuous-time state equations using Backward Euler integration
def f_discrete_implicit(
    x,
    u,
    theta,
    dt,
    model_option=None,
):
    n = len(x)

    f0 = f_continuous(
        np.zeros(n),
        u,
        theta,
        model_option=model_option,
    )

    A = np.zeros((n, n))

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0

        A[:, i] = (
            f_continuous(
                e,
                u,
                theta,
                model_option=model_option,
            )
            - f0
        )

    return np.linalg.solve(
        np.eye(n) - dt * A,
        x + dt * f0,
    )

def simulate_model(df, theta):
    """Simulate the selected SDE model over a dataframe."""
    data = df.copy()
    data[cfg.TS_COL] = pd.to_datetime(data[cfg.TS_COL], errors="coerce", utc=True)

    dt = data[cfg.TS_COL].diff().dt.total_seconds().median()

    inputs = data[sde_cfg.INPUT_COLS].to_numpy(dtype=float)

    states = np.zeros((len(data), len(sde_cfg.STATE_COLS),),dtype=float)

    states[0] = (
        data[sde_cfg.STATE_COLS]
        .iloc[0]
        .to_numpy(dtype=float)
    )

    for k in range(len(data) - 1):
        states[k + 1] = f_discrete_implicit(x=states[k], u=inputs[k], theta=theta, dt=dt)
    return states