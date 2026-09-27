from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

import config as cfg
from Methods.Blackbox.common_BB_scripts import set_reproducibility
from Methods.Blackbox.profile_dataset import (
    create_profiles,
    scale_train_val_test_data,
    tensorize_profiles,
)
from Methods.TNN.experiment_configs import (
    TNN_MODEL_TYPE,
    TNN_TRAINING_CONFIG,
    WINDOW_STEPS,
)
from Methods.TNN.tnn_model import build_model
from scripts.Visualization.metrics import compute_metrics
from scripts.misc.feature_engineering import ts_cols


def load_and_prepare_data(
    csv_path: Path,
) -> tuple[pd.DataFrame, float]:
    """Load one pre-split dataset and map DS-specific names to generic names."""

    data = pd.read_csv(csv_path)
    data, dt_s = ts_cols(data, cfg.TS_COL)

    required_generic = set(
        [*cfg.INPUT_COLS, *cfg.TARGET_COLS]
    )

    raw_model_cols = [
        raw_col
        for raw_col, generic_col in cfg.RENAME_MAP.items()
        if generic_col in required_generic
    ]

    required_raw_cols = [cfg.TS_COL, *raw_model_cols]

    missing = [
        col
        for col in required_raw_cols
        if col not in data.columns
    ]

    if missing:
        raise KeyError(
            f"Missing required columns in {csv_path}: {missing}"
        )

    data = data[required_raw_cols].copy()
    data = data.rename(columns=cfg.RENAME_MAP)

    return data, dt_s


def get_tnn_column_groups() -> tuple[list[str], list[str]]:
    temperature_inputs = [
        col
        for col in cfg.INPUT_COLS
        if col.startswith("T")
    ]

    temperature_cols = list(
        dict.fromkeys(
            [*cfg.TARGET_COLS, *temperature_inputs]
        )
    )

    # T16 = FAN1 .
    cooling_columns = ["T16"] if "T16" in cfg.INPUT_COLS else []

    return temperature_cols, cooling_columns


def _one_step_views(
    sequence_tensor: torch.Tensor,
    sequence_mask: torch.Tensor,
    n_in: int,
    n_out: int,
):
    """Return u(k), y(k+1), valid-pair mask and initial y(k)."""

    if sequence_tensor.shape[1] < 2:
        raise ValueError(
            "A TNN profile must contain at least two time steps."
        )

    x_seq = sequence_tensor[:, :-1, :n_in]
    y_next = sequence_tensor[:, 1:, -n_out:]

    pair_mask = (
        sequence_mask[:, :-1]
        & sequence_mask[:, 1:]
    )

    state0 = sequence_tensor[:, 0, -n_out:]

    return x_seq, y_next, pair_mask, state0


def masked_mse(
    y_hat: torch.Tensor,
    y_true: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).to(
        dtype=y_hat.dtype,
        device=y_hat.device,
    ).expand_as(y_hat)

    err = (y_hat - y_true).pow(2) * mask_f
    return err.sum() / mask_f.sum().clamp(min=1.0)


def train_tnn(
    model: nn.Module,
    train_tensor: torch.Tensor,
    train_mask: torch.Tensor,
    val_tensor: torch.Tensor,
    val_mask: torch.Tensor,
    input_cols: list[str],
    target_cols: list[str],
    dt_s: float,
    *,
    n_epochs: int,
    tbptt_size: int,
    lr: float,
    weight_decay: float,
    smoothness_weight: float,
):
    """Train on TRAIN and restore the epoch with the lowest VALIDATION MSE."""

    n_in = len(input_cols)
    n_out = len(target_cols)

    x_train, y_train, m_train, state0_train = _one_step_views(
        train_tensor,
        train_mask,
        n_in,
        n_out,
    )

    x_val, y_val, m_val, state0_val = _one_step_views(
        val_tensor,
        val_mask,
        n_in,
        n_out,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    loss_func = nn.MSELoss(reduction="none")

    best_val_loss = float("inf")
    best_state = None
    best_epoch = None

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    total_steps = x_train.shape[1]
    n_chunks = int(np.ceil(total_steps / tbptt_size))

    for epoch in range(1, n_epochs + 1):
        model.train()
        hidden = state0_train
        epoch_loss = 0.0

        for chunk_idx in range(n_chunks):
            t0 = chunk_idx * tbptt_size
            t1 = min(
                (chunk_idx + 1) * tbptt_size,
                total_steps,
            )

            x = x_train[:, t0:t1, :]
            y = y_train[:, t0:t1, :]
            m = m_train[:, t0:t1]

            optimizer.zero_grad(set_to_none=True)

            y_hat, hidden = model(
                x,
                hidden.detach(),
            )

            mask_f = m.unsqueeze(-1).to(
                dtype=y_hat.dtype,
                device=y_hat.device,
            )

            point_loss = loss_func(y_hat, y) * mask_f
            denom = (
                mask_f.sum()
                * n_out
            ).clamp(min=1.0)
            prediction_loss = point_loss.sum() / denom

            if y_hat.shape[1] > 1:
                transition_mask = m[:, 1:] & m[:, :-1]
                transition_mask_f = transition_mask.unsqueeze(-1).to(
                    dtype=y_hat.dtype,
                    device=y_hat.device,
                )

                dy = (
                    y_hat[:, 1:] - y_hat[:, :-1]
                ) / dt_s

                trend_denom = (
                    transition_mask_f.sum()
                    * n_out
                ).clamp(min=1.0)

                trend_penalty = (
                    dy.pow(2) * transition_mask_f
                ).sum() / trend_denom
            else:
                trend_penalty = y_hat.new_tensor(0.0)

            loss = (
                prediction_loss
                + smoothness_weight * trend_penalty
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )
            optimizer.step()

            epoch_loss += float(loss.item())

        train_loss = epoch_loss / max(n_chunks, 1)
        history["train_loss"].append(train_loss)

        model.eval()
        with torch.no_grad():
            y_val_hat, _ = model(x_val, state0_val)
            val_loss = masked_mse(
                y_val_hat,
                y_val,
                m_val,
            ).item()

        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f}"
        )

    if best_state is None:
        raise RuntimeError("No validation checkpoint was produced.")

    model.load_state_dict(best_state)

    print(
        f"Restored best validation epoch {best_epoch} "
        f"with loss {best_val_loss:.6f}"
    )

    return model, history, best_epoch, best_val_loss


def predict_tnn(
    model: nn.Module,
    sequence_tensor: torch.Tensor,
    sequence_mask: torch.Tensor,
    input_cols: list[str],
    target_cols: list[str],
    y_scaler,
):
    """Generate one-step-ahead predictions in physical temperature units."""

    n_in = len(input_cols)
    n_out = len(target_cols)

    x_seq, y_next, pair_mask, state0 = _one_step_views(
        sequence_tensor,
        sequence_mask,
        n_in,
        n_out,
    )

    model.eval()
    with torch.no_grad():
        pred_scaled, _ = model(x_seq, state0)

    pred_scaled_np = pred_scaled.detach().cpu().numpy()
    batch_size, seq_len, n_features = pred_scaled_np.shape

    pred_c = y_scaler.inverse_transform(
        pred_scaled_np.reshape(-1, n_features)
    ).reshape(batch_size, seq_len, n_features)

    return pred_c, y_next, pair_mask


def evaluate_tnn(
    model: nn.Module,
    test_tensor: torch.Tensor,
    test_mask: torch.Tensor,
    input_cols: list[str],
    target_cols: list[str],
    y_scaler,
):
    pred_c, y_true_scaled, pair_mask = predict_tnn(
        model,
        test_tensor,
        test_mask,
        input_cols,
        target_cols,
        y_scaler,
    )

    y_true_scaled_np = y_true_scaled.detach().cpu().numpy()
    batch_size, seq_len, n_features = y_true_scaled_np.shape

    y_true_c = y_scaler.inverse_transform(
        y_true_scaled_np.reshape(-1, n_features)
    ).reshape(batch_size, seq_len, n_features)

    mask_np = pair_mask.detach().cpu().numpy().astype(bool)

    metrics = compute_metrics(
        y_true_c[mask_np],
        pred_c[mask_np],
    )

    return metrics


def save_tnn(
    model: nn.Module,
    x_scaler,
    y_scaler,
    input_cols: list[str],
    target_cols: list[str],
    temperature_cols: list[str],
    cooling_columns: list[str],
    dt_s: float,
    history: dict,
    best_epoch: int,
    best_val_loss: float,
    path: str | Path,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "model_type": TNN_MODEL_TYPE,
        "dataset": cfg.DS,
        "experiment_config": cfg.CONFIG,
        "input_cols": list(input_cols),
        "target_cols": list(target_cols),
        "temperature_cols": list(temperature_cols),
        "cooling_columns": list(cooling_columns),
        "dt_s": float(dt_s),
        "window_steps": WINDOW_STEPS,
        "training_config": dict(TNN_TRAINING_CONFIG),
        "n_neurons": TNN_TRAINING_CONFIG["n_neurons"],
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "one_step_ahead": True,
    }

    torch.save(checkpoint, path)
    print(f"TNN model saved to {path}")


def main():
    set_reproducibility(cfg.SEED)

    data_path = Path(cfg.DATA_PATH)
    model_config = cfg.CONFIG
    # -----------------------------------------------------
    # Load the three pre-split datasets
    # -----------------------------------------------------
    train_path = data_path/"train_test_split"/ f"{cfg.DS}_generator_train.csv"
    val_path = data_path/"train_test_split"/f"{cfg.DS}_generator_val.csv"
    test_path = data_path/"train_test_split"/ f"{cfg.DS}_generator_test.csv"

    train_data, train_dt_s = load_and_prepare_data(train_path)
    val_data, val_dt_s = load_and_prepare_data(val_path)
    test_data, test_dt_s = load_and_prepare_data(test_path)

    dt_s = train_dt_s

    # -----------------------------------------------------
    # Profile each split independently
    # -----------------------------------------------------
    train_data, train_profiles = create_profiles(
        train_data,
        ts_col=cfg.TS_COL,
        window_steps=WINDOW_STEPS,
        dt_s=train_dt_s,
    )
    val_data, val_profiles = create_profiles(
        val_data,
        ts_col=cfg.TS_COL,
        window_steps=WINDOW_STEPS,
        dt_s=val_dt_s,
    )
    test_data, test_profiles = create_profiles(
        test_data,
        ts_col=cfg.TS_COL,
        window_steps=WINDOW_STEPS,
        dt_s=test_dt_s,
    )
    # -----------------------------------------------------
    # Fit scalers on training data and apply to all splits
    # -----------------------------------------------------
    (
        train_data,
        val_data,
        test_data,
        x_scaler,
        y_scaler,
    ) = scale_train_val_test_data(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        input_cols=cfg.INPUT_COLS,
        target_cols=cfg.TARGET_COLS,
    )
    # -----------------------------------------------------
    # Tensor creation
    # -----------------------------------------------------
    x_train, y_train, train_mask = tensorize_profiles(
        train_data,
        train_profiles,
        cfg.INPUT_COLS,
        cfg.TARGET_COLS,
        device=cfg.DEVICE,
    )
    x_val, y_val, val_mask = tensorize_profiles(
        val_data,
        val_profiles,
        cfg.INPUT_COLS,
        cfg.TARGET_COLS,
        device=cfg.DEVICE,
    )
    x_test, y_test, test_mask = tensorize_profiles(
        test_data,
        test_profiles,
        cfg.INPUT_COLS,
        cfg.TARGET_COLS,
        device=cfg.DEVICE,
    )

    train_tensor = torch.cat([x_train, y_train], dim=2)
    val_tensor = torch.cat([x_val, y_val], dim=2)
    test_tensor = torch.cat([x_test, y_test], dim=2)

    temperature_cols, cooling_columns = get_tnn_column_groups()

    print("Dataset:", cfg.DS)
    print("Configuration:", cfg.CONFIG)
    print("Inputs:", cfg.INPUT_COLS)
    print("Targets:", cfg.TARGET_COLS)
    print("Thermal nodes:", temperature_cols)
    print("Cooling columns:", cooling_columns)
    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = build_model(
        dt_s=dt_s,
        input_cols=cfg.INPUT_COLS,
        target_cols=cfg.TARGET_COLS,
        temperature_cols=temperature_cols,
        device=cfg.DEVICE,
        cooling_columns=cooling_columns,
        n_neurons=TNN_TRAINING_CONFIG["n_neurons"],
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    model, history, best_epoch, best_val_loss = train_tnn(
        model=model,
        train_tensor=train_tensor,
        train_mask=train_mask,
        val_tensor=val_tensor,
        val_mask=val_mask,
        input_cols=cfg.INPUT_COLS,
        target_cols=cfg.TARGET_COLS,
        dt_s=dt_s,
        **{
            key: value
            for key, value in TNN_TRAINING_CONFIG.items()
            if key != "n_neurons"
        },
    )
    # -----------------------------------------------------
    # Final test evaluation
    # -----------------------------------------------------
    metrics = evaluate_tnn(
        model=model,
        test_tensor=test_tensor,
        test_mask=test_mask,
        input_cols=cfg.INPUT_COLS,
        target_cols=cfg.TARGET_COLS,
        y_scaler=y_scaler,
    )

    print(
        "Final test metrics: "
        f"rmse={metrics['rmse']:.4f}, "
        f"mae={metrics['mae']:.4f}, "
        f"mse={metrics['mse']:.4f}, "
        f"max_abs={metrics['max_abs']:.4f}"
    )

    model_path = f"{cfg.DS}_{cfg.OUTPUT_TYPE}_{model_config}_tnn_winding_baseline.pt"

    # -----------------------------------------------------
    # Save model and preprocessing metadata
    # -----------------------------------------------------
    save_tnn(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        input_cols=cfg.INPUT_COLS,
        target_cols=cfg.TARGET_COLS,
        temperature_cols=temperature_cols,
        cooling_columns=cooling_columns,
        dt_s=dt_s,
        history=history,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        path=model_path,
    )


if __name__ == "__main__":
    main()
