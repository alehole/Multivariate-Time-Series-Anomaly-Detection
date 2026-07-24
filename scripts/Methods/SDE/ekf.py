import numpy as np
from model import f_discrete_implicit
from config import STATE_COLS, MEAS_COLS, INPUT_COLS


def jacobian_F(x, u_k, theta, dt, eps=1e-5):
    n = len(x)
    F = np.zeros((n, n))
    fx = f_discrete_implicit(x, u_k, theta, dt)

    for i in range(n):
        x_eps = x.copy()
        x_eps[i] += eps
        fx_eps = f_discrete_implicit(x_eps, u_k, theta, dt)
        F[:, i] = (fx_eps - fx) / eps

    return F


def run_ekf(df, theta, C, Q, R):
    """
        ----------------------------------
        theta       : model parameter vector θ, e.g. θ = [Cw, Rg]
        x_pred      : prior state estimate,        x̂_{k+1|k}
        x_hat       : posterior state estimate,    x̂_{k|k}
        P_pred      : prior covariance,            P_{k+1|k}
        P_cov       : posterior covariance,        P_{k|k}
        F           : state transition Jacobian,   F_k = ∂f/∂x
        C           : measurement/output matrix,   often denoted H_k in EKF literature
        Q           : process noise covariance
        R           : measurement noise covariance
        innovation  : prediction residual,         ε_k = y_k - ŷ_{k|k-1}
        S           : innovation covariance,       S_k = C P_{k|k-1} Cᵀ + R
        K           : Kalman gain,                 K_k

        NIS         : normalized innovation squared,
                      ε_kᵀ S_k⁻¹ ε_k
        neg_log_lik : negative log-likelihood cost,
                      g(θ) = Σ[ε_kᵀ S_k⁻¹ ε_k + log(det(S_k))]
    """


    dt = df["Created"].diff().dt.total_seconds().median()

    Y = df[MEAS_COLS].values
    U = df[INPUT_COLS].values

    n_states = len(STATE_COLS)
    n_meas = len(MEAS_COLS)
    N = len(df)

    x_pred_hist = np.zeros((N, n_states))
    x_hat = np.zeros((N, n_states))
    P_cov = np.zeros((N, n_states, n_states))
    innovations = np.zeros((N, n_meas))
    NIS = np.zeros(N)

    x_hat[0] = df[STATE_COLS].iloc[0].values
    x_pred_hist[0] = x_hat[0]
    P_cov[0] = np.eye(n_states)

    I = np.eye(n_states)
    neg_log_lik  = 0.0
    for k in range(N - 1):
        u_k = U[k, 0] if U.ndim == 2 and U.shape[1] == 1 else U[k]

        # =====================================================
        # 1. PREDICTION
        # =====================================================

        x_pred = f_discrete_implicit(x_hat[k], u_k, theta, dt) # Prediction
        x_pred_hist[k + 1] = x_pred

        # Linearize the nonlinear transition model:
        # F_k = ∂f/∂x
        F = jacobian_F(x_hat[k], u_k, theta, dt)

        # Predict the state covariance:
        # P_{k+1|k} = F_k P_{k|k} F_kᵀ + Q
        P_pred = F @ P_cov[k] @ F.T + Q # Covariance

        # =====================================================
        # 2. MEASUREMENT PREDICTION
        # =====================================================

        # Predict what the measurement should be:
        # ŷ_{k+1|k} = C x̂_{k+1|k}
        y_meas = Y[k + 1]
        y_pred = C @ x_pred

        # Difference between the actual and predicted measurement:
        # ε_{k+1} = y_{k+1} - ŷ_{k+1|k}
        innovation = y_meas - y_pred

        # Innovation covariance:
        # S_{k+1} = C P_{k+1|k} Cᵀ + R
        S = C @ P_pred @ C.T + R        # innovation covariance

        # Parameter estimation
        nis_k = innovation.T @ np.linalg.solve(S, innovation) # NIS
        log_det_S = np.linalg.slogdet(S)[1]
        neg_log_lik  += nis_k + log_det_S # Negative log-likelihood Eq. (14)

        # =====================================================
        # 3. CORRECTION / UPDATE
        # =====================================================

        # Compute the Kalman gain:
        # K_{k+1} = P_{k+1|k} Cᵀ S_{k+1}^{-1}
        K = np.linalg.solve(S.T, (P_pred @ C.T).T).T # Kalman gain

        # Correct the predicted state:
        # x̂_{k+1|k+1} = x̂_{k+1|k} + K_{k+1} ε_{k+1}
        x_hat[k + 1] = x_pred + K @ innovation # Posterior

        # Correct the covariance using the Joseph form:
        # P_{k+1|k+1}
        # = (I-KC)P_{k+1|k}(I-KC)ᵀ + KRKᵀ
        A_ = I - K @ C
        P_cov[k + 1] = A_ @ P_pred @ A_.T + K @ R @ K.T

        innovations[k + 1] = innovation
        NIS[k + 1] = nis_k

    return x_hat, P_cov, innovations, NIS, x_pred_hist, neg_log_lik