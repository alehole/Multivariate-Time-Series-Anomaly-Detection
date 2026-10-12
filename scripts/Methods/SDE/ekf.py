import numpy as np
from SDE_config import STATE_COLS, MEAS_COLS, INPUT_COLS, MODEL_OPTION
from thermal_models import discrete_state_space

def run_kf(
    df,
    theta,
    H,
    Q,
    R,
    state_cols=None,
    meas_cols=None,
    input_cols=None,
    model_option=None,
):
    """
    Standard Kalman filter for the linear thermal state-space model.

    theta       : model parameter vector
    F           : discrete state-transition matrix
    G           : discrete input matrix
    H           : measurement/output matrix
    Q           : process-noise covariance
    R           : measurement-noise covariance

    x_pred      : prior state estimate,       x̂_{k+1|k}
    x_hat       : posterior state estimate,   x̂_{k|k}
    P_pred      : prior covariance,           P_{k+1|k}
    P_cov       : posterior covariance,       P_{k|k}

    innovation  : prediction residual,
                  ε_k = y_k - ŷ_{k|k-1}

    S           : innovation covariance,
                  S_k = H P_{k|k-1} Hᵀ + R

    K           : Kalman gain

    NIS         : normalized innovation squared,
                  ε_kᵀ S_k⁻¹ ε_k

    neg_log_lik : likelihood objective,
                  g(θ) = Σ[ε_kᵀ S_k⁻¹ ε_k + log(det(S_k))]
    """
    if state_cols is None:
        state_cols = STATE_COLS

    if meas_cols is None:
        meas_cols = MEAS_COLS

    if input_cols is None:
        input_cols = INPUT_COLS

    if model_option is None:
        model_option = MODEL_OPTION

    dt = df["Created"].diff().dt.total_seconds().median()

    Y = df[meas_cols].values
    U = df[input_cols].values

    n_states = len(state_cols)
    n_meas = len(meas_cols)
    N = len(df)

    x_pred_hist = np.zeros((N, n_states))
    x_hat = np.zeros((N, n_states))
    P_cov = np.zeros((N, n_states, n_states))
    innovations = np.zeros((N, n_meas))
    NIS = np.zeros(N)

    # Initial state
    x_hat[0] = df[state_cols].iloc[0].values
    x_pred_hist[0] = x_hat[0]

    # Initial state covariance
    P_cov[0] = np.eye(n_states)

    I = np.eye(n_states)

    neg_log_lik  = 0.0

    # -----------------------------------------------------
    # Linear discrete state-space matrices
    # -----------------------------------------------------
    F, G = discrete_state_space(
        theta=theta,
        dt=dt,
        model_option=model_option,
    )

    # -----------------------------------------------------
    # Kalman filter
    # -----------------------------------------------------
    for k in range(N - 1):

        u_k = U[k, 0] if U.ndim == 2 and U.shape[1] == 1 else U[k]

        # =====================================================
        # 1. PREDICTION
        # =====================================================

        # x̂_{k+1|k} = F x̂_{k|k} + G u_k
        x_pred = F @ x_hat[k] + G @ u_k

        x_pred_hist[k + 1] = x_pred

        # Predict the state covariance:
        # P_{k+1|k} = F P_{k|k} Fᵀ + Q
        P_pred = F @ P_cov[k] @ F.T + Q

        # =====================================================
        # 2. MEASUREMENT PREDICTION
        # =====================================================

        y_meas = Y[k + 1]

        # Predict what the measurement should be:
        # ŷ_{k+1|k} = H x̂_{k+1|k}
        y_pred = H @ x_pred

        # Difference between the actual and predicted measurement:
        # ε_{k+1} = y_{k+1} - ŷ_{k+1|k}
        innovation = y_meas - y_pred

        # Innovation covariance:
        # S_{k+1} = H P_{k+1|k} Hᵀ + R
        S = H @ P_pred @ H.T + R

        # =================================================
        # 3. LIKELIHOOD
        # ================================================

        nis_k = innovation.T @ np.linalg.solve(S, innovation) # NIS
        log_det_S = np.linalg.slogdet(S)[1]
        neg_log_lik  += nis_k + log_det_S # Negative log-likelihood Eq. (14)

        # =====================================================
        # 3. CORRECTION / UPDATE
        # =====================================================

        # Compute the Kalman gain:
        # K_{k+1} = P_{k+1|k} Cᵀ S_{k+1}^{-1}
        K = np.linalg.solve(S.T, (P_pred @ H.T).T).T # Kalman gain

        # Correct the predicted state:
        # x̂_{k+1|k+1} = x̂_{k+1|k} + K_{k+1} ε_{k+1}
        x_hat[k + 1] = x_pred + K @ innovation # Posterior

        # Correct the covariance using the Joseph form:
        # P_{k+1|k+1}
        # = (I-KC)P_{k+1|k}(I-KC)ᵀ + KRKᵀ
        A_ = I - K @ H
        P_cov[k + 1] = A_ @ P_pred @ A_.T + K @ R @ K.T

        innovations[k + 1] = innovation
        NIS[k + 1] = nis_k

    return x_hat, P_cov, innovations, NIS, x_pred_hist, neg_log_lik