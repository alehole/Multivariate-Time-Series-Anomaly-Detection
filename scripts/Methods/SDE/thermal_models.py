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

    dT = P / C1 + (Tref - T) / (C1 * R1)

    return np.array([dT], dtype=float)


def model_1state_2ref(
    x: Array,
    u: Array,
    theta: Array,
) -> Array:
    """
                    P
                    ↓
                [T1, C1]
                 │      │
                Ra      Rc
                 │      │
               [Ta]   [Tc]
    """
    T1 = x[0]
    P, Ta, Tc = u
    C1, Ra, Rc = theta

    q_a = (T1 - Ta) / Ra
    q_c = (T1 - Tc) / Rc

    dT1 = (P - q_a - q_c) / C1

    return np.array([dT1], dtype=float)


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

    q_12 = (T1 - T2) / R1
    q_2ref = (T2 - Tref) / R2

    dT1 = (P - q_12) / C1
    dT2 = (q_12 - q_2ref) / C2

    return np.array([dT1, dT2], dtype=float)


MODEL_FUNCTIONS: dict[str, ModelFunction] = {
    "1state": model_1state,
    "1state_2ref": model_1state_2ref,
    "2state": model_2state,
    "2state_A": model_2state,
    "2state_B": model_2state,
}


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