from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]
ModelFunction = Callable[[Array, Array, Array], Array]


def model_1state(
    x: Array,
    u: Array,
    theta: Array,
) -> Array:
    """
                P
                ↓
            [T1, C1]
                │
               R1
                │
              [Tref]
    """
    T = x[0]
    P, Tref = u
    C1, R1 = theta

    Kp = 1.0
    q_heat = Kp * P

    dT = q_heat / C1 + (Tref - T) / (C1 * R1)

    return np.array([dT], dtype=float)

def state_space_1state(theta: Array) -> tuple[Array, Array]:
    C1, R1 = theta

    A = np.array([
        [-1.0 / (C1 * R1)]
    ])

    B = np.array([
        [1.0 / C1, 1.0 / (C1 * R1)]
    ])

    return A, B

def model_2state(
    x: Array,
    u: Array,
    theta: Array,
) -> Array:
    """
                P
                ↓
            [T1, C1]
                │
               R1
                │
            [T2, C2]
                │
               R2
                │
             [Tref]
    """
    T1, T2 = x
    P, Tref = u
    C1, C2, R1, R2 = theta

    Kp = 1.0
    q_heat = Kp * P

    q_12 = (T1 - T2) / R1
    q_2ref = (T2 - Tref) / R2

    #dT1 = (P - q_12) / C1
    dT1 = (q_heat - q_12) / C1
    dT2 = (q_12 - q_2ref) / C2

    return np.array([dT1, dT2], dtype=float)


def state_space_2state(theta: Array) -> tuple[Array, Array]:
    C1, C2, R1, R2 = theta

    A = np.array([
        [
            -1.0 / (C1 * R1),
             1.0 / (C1 * R1),
        ],
        [
             1.0 / (C2 * R1),
            -(1.0 / (C2 * R1) + 1.0 / (C2 * R2)),
        ],
    ])

    B = np.array([
        [
            1.0 / C1,
            0.0,
        ],
        [
            0.0,
            1.0 / (C2 * R2),
        ],
    ])

    return A, B

MODEL_FUNCTIONS: dict[str, ModelFunction] = {
    "1state": model_1state,
    "2state": model_2state,
    "2state_A": model_2state,
    "2state_B": model_2state,
}
STATE_SPACE_FUNCTIONS = {
    "1state": state_space_1state,
    "2state": state_space_2state,
    "2state_A": state_space_2state,
    "2state_B": state_space_2state,
}

def continuous_state_space(
    theta,
    model_option: str,
) -> tuple[Array, Array]:

    try:
        state_space_function = STATE_SPACE_FUNCTIONS[model_option]
    except KeyError as exc:
        raise ValueError(
            f"Invalid model option {model_option!r}. "
            f"Expected one of {list(STATE_SPACE_FUNCTIONS)}."
        ) from exc

    theta = np.asarray(theta, dtype=float).ravel()

    return state_space_function(theta)

def f_continuous(
    x,
    u,
    theta,
    model_option: str,
) -> Array:
    """Evaluate the selected continuous-time thermal model."""
    try:
        model_function = MODEL_FUNCTIONS[model_option]
    except KeyError as exc:
        raise ValueError(
            f"Invalid model option {model_option!r}. "
            f"Expected one of {list(MODEL_FUNCTIONS)}."
        ) from exc

    x_array = np.asarray(x, dtype=float).ravel()
    u_array = np.asarray(u, dtype=float).ravel()
    theta_array = np.asarray(theta, dtype=float).ravel()

    return model_function(
        x_array,
        u_array,
        theta_array,
    )

def discrete_state_space(
    theta,
    dt: float,
    model_option: str,
) -> tuple[Array, Array]:

    A_c, B_c = continuous_state_space(
        theta,
        model_option,
    )

    I = np.eye(A_c.shape[0])

    M = I - dt * A_c

    F = np.linalg.solve(M, I)
    G = np.linalg.solve(M, dt * B_c)

    return F, G