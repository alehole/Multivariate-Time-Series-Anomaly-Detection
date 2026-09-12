from pathlib import Path
from itertools import product
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor

from src.models.profile_dataset import (
    train_val_test_split_profiles,
    scale_profile_data,
    tensorize_profiles,
)

from src.models.model_utils import (
    evaluate_predictions,
    print_metrics,
    eval_plot_from_predictions,
    set_seed,
)

from src.config import paths
from src.models.TNN.tnn_model import build_model
from src.preprocess.feature_engineering import ts_cols, feature_processing

torch.set_num_threads(os.cpu_count() or 1)
torch.set_num_interop_threads(1)
print(torch.get_num_threads())

#from src.models.TNN.experiment_configs import CYL3_CONFIG as cfg
from src.models.TNN.experiment_configs import CYL3_CONFIG as cfg
#from src.models.TNN.experiment_configs import CYL_all_CONFIG as cfg
#from src.models.TNN.experiment_configs import WINDING_CONFIG_1 as cfg

# ============================================================
# COLUMN GROUPS
# ============================================================
def get_column_groups(
    df: pd.DataFrame,
    target_cols: list[str] | None = None,
    temperature_cols: list[str] | None = None,
    drop_cols: list[str] | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:

    target_cols = [c for c in target_cols if c in df.columns]

    if not target_cols:
        raise ValueError("No target columns found in dataset")

    temperature_cols = [c for c in temperature_cols if c in df.columns]

    drop_columns = drop_cols
    drop_columns = [c for c in drop_columns if c in df.columns]

    # Non-temperature inputs
    non_temperature_cols = [
        c for c in df.columns
        if c not in set(temperature_cols + drop_columns)
    ]

    # Temperature inputs excluding targets
    extra_temp_cols = [c for c in temperature_cols if c not in target_cols]

    # Final ordered input columns
    input_cols = extra_temp_cols + non_temperature_cols

    return target_cols, temperature_cols, non_temperature_cols, input_cols


def train_model(
    model: nn.Module,
    train_tensor: Tensor,
    train_mask: Tensor,
    input_cols: list[str],
    target_cols: list[str],
    dt_s: float,
    *,
    n_epochs: int = 200,
    tbptt_size: int = 512,
    lr: float = 1e-2,
    weight_decay: float = 1e-5,
    smoothness_weight: float = 0.01,
) -> nn.Module:

    # Pointwise MSE is used so that padding can be masked out manually later
    loss_func = nn.MSELoss(reduction="none")

    # Adam optimizer for parameter updates
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Number of predicted temperatures and number of input features
    n_out = len(target_cols)
    n_in = len(input_cols)

    # Tensor shape:
    # B = number of training profiles
    # T_total = total number of timesteps per padded sequence
    B, T_total, _ = train_tensor.shape

    # Number of temporal chunks used for truncated backpropagation
    n_batches = int(np.ceil(T_total / tbptt_size))

    print(f"Train tensor shape: {train_tensor.shape}")
    print(f"Train mask shape  : {train_mask.shape}")
    print(f"B={B}, T={T_total}, n_in={n_in}, n_out={n_out}, chunks={n_batches}")

    # Put model in training mode
    model.train()

    with tqdm(desc="Training", total=n_epochs) as pbar:
        for epoch in range(n_epochs):

            # Initialize hidden thermal state using the first true target value
            # Shape: (B, n_out)
            hidden = train_tensor[:, 0, -n_out:]

            # Store average epoch loss
            epoch_loss = 0.0

            for i in range(n_batches):
                # Time range for current truncated BPTT chunk
                t0 = i * tbptt_size
                t1 = min((i + 1) * tbptt_size, T_total)

                # Reset gradients before backpropagation
                opt.zero_grad(set_to_none=True)

                # Slice current time chunk
                # x: input features
                # y: target temperatures
                # m: valid timestep mask
                x = train_tensor[:, t0:t1, :n_in]     # (B, Tchunk, n_in)
                y = train_tensor[:, t0:t1, -n_out:]   # (B, Tchunk, n_out)
                m = train_mask[:, t0:t1]              # (B, Tchunk)

                # Forward pass through model
                # hidden.detach() breaks gradient flow between chunks
                # and implements truncated backpropagation through time
                yhat, hidden = model(x, hidden.detach())   # (B, Tchunk, n_out)

                # ------------------------------------------------------------
                # 1) Masked mean squared error
                # ------------------------------------------------------------
                # Compute pointwise squared error
                loss_pt = loss_func(yhat, y)  # (B, Tchunk, n_out)

                # Ignore padded timesteps using the mask
                loss_pt = loss_pt * m[:, :, None]

                # Normalize by number of valid timesteps and targets
                denom = m.sum().clamp(min=1).float()
                loss_mse = loss_pt.sum() / (denom * n_out)

                # ------------------------------------------------------------
                # 2) Trend / smoothness penalty
                # ------------------------------------------------------------
                # Only compute temporal derivatives where two consecutive
                # timesteps are both valid
                m_tr = m[:, 1:] & m[:, :-1]   # (B, Tchunk-1)

                # Approximate temperature derivative dT/dt
                dy = (yhat[:, 1:] - yhat[:, :-1]) / dt_s

                # Penalize large temperature rate-of-change to encourage
                # smoother and more physically realistic thermal behavior
                trend_penalty = (
                    dy.pow(2) * m_tr[:, :, None]
                ).sum() / (m_tr.sum().clamp(min=1).float() * n_out)

                # Total loss = prediction loss + smoothness regularization
                loss = loss_mse + smoothness_weight * trend_penalty

                # Backpropagation
                loss.backward()

                # Gradient clipping improves stability for recurrent training
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update model parameters
                opt.step()

                # Accumulate epoch loss
                epoch_loss += float(loss.item())

            # Average loss over all chunks
            epoch_loss /= n_batches

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix_str(f"loss: {epoch_loss:.2e}")

    return model


def predict_test_set(
    model: nn.Module,
    test_tensor: Tensor,
    input_cols: list[str],
    target_cols: list[str],
    y_scaler,
) -> np.ndarray:

    # Put model in evaluation mode (disables dropout, etc.)
    model.eval()

    # Number of input features and target variables
    n_in = len(input_cols)
    n_out = len(target_cols)

    # Disable gradient computation during inference
    with torch.no_grad():

        # Extract model inputs
        # Shape: (B, T, n_in)
        x_test = test_tensor[:, :, :n_in]

        # Initial thermal state taken from the first true target value
        # Shape: (B, n_out)
        state0 = test_tensor[:, 0, -n_out:]

        # Forward pass through the model
        # pred shape: (B, T, n_out)
        pred, _ = model(x_test, state0)

        # Move tensor to CPU and convert to NumPy
        pred = pred.cpu().numpy()

    # Extract dimensions
    B, T, S = pred.shape

    # ------------------------------------------------------------
    # Convert predictions back to original temperature scale
    # ------------------------------------------------------------
    # During training targets were scaled with. RobustScaler).
    # Here we apply the inverse transform to recover °C values.
    pred_c = y_scaler.inverse_transform(
        pred.reshape(-1, S)
    ).reshape(B, T, S)

    return pred_c


# ============================================================
# SAVE MODEL
# ============================================================
def save_model(
    model: nn.Module,
    x_scaler,
    y_scaler,
    input_cols: list[str],
    target_cols: list[str],
    temperature_cols: list[str],
    dt_s: float,
    n_neurons: int,
    cooling_columns: list[str],
    path: str ,
    best_params: dict | None = None,
):

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "model_state_dict": model.state_dict(),
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "input_cols": input_cols,
        "target_cols": target_cols,
        "temperature_cols": temperature_cols,
        "dt_s": dt_s,
        "n_neurons": n_neurons,
        "cooling_columns": cooling_columns,
    }
    if best_params is not None:
        save_dict["best_params"] = best_params

    torch.save(save_dict, path)
    print(f"Model saved to {path}")

def run_grid_search(
    grid: dict,
    dt_s: float,
    input_cols: list[str],
    target_cols: list[str],
    temperature_cols: list[str],
    cooling_columns: list[str],
    train_tensor: Tensor,
    train_mask: Tensor,
    val_tensor: Tensor,
    val_mask: Tensor,
    y_scaler,
    device: torch.device,
) -> tuple[nn.Module, dict, float]:

    best_score = float("inf")
    best_params = None
    best_model = None

    total_runs = (
        len(grid["n_epochs"])
        * len(grid["lr"])
        * len(grid["tbptt_size"])
        * len(grid["weight_decay"])
        * len(grid["smoothness_weight"])
        * len(grid["n_neurons"])
    )

    for i, (n_epochs, lr, tbptt_size, wd, smooth, n_neurons) in enumerate(
        product(
            grid["n_epochs"],
            grid["lr"],
            grid["tbptt_size"],
            grid["weight_decay"],
            grid["smoothness_weight"],
            grid["n_neurons"],
        ),
        start=1,
    ):
        print(
            f"\n[{i}/{total_runs}] ({i / total_runs * 100:.1f}%) "
            f"\nTRAINING WITH PARAMETERS: "
            f"n_epochs={n_epochs}, lr={lr}, tbptt_size={tbptt_size}, "
            f"weight_decay={wd}, smoothness_weight={smooth}, "
            f"n_neurons={n_neurons}"
        )

        model = build_model(
            dt_s=dt_s,
            input_cols=input_cols,
            target_cols=target_cols,
            temperature_cols=temperature_cols,
            device=device,
            cooling_columns=cooling_columns,
            n_neurons=n_neurons,
        )

        model = train_model(
            model=model,
            train_tensor=train_tensor,
            train_mask=train_mask,
            input_cols=input_cols,
            target_cols=target_cols,
            dt_s=dt_s,
            n_epochs=n_epochs,
            lr=lr,
            tbptt_size=tbptt_size,
            weight_decay=wd,
            smoothness_weight=smooth,
        )

        pred_c = predict_test_set(
            model=model,
            test_tensor=val_tensor,
            input_cols=input_cols,
            target_cols=target_cols,
            y_scaler=y_scaler,
        )

        metrics = evaluate_predictions(
            pred_c=pred_c,
            y_true_scaled=val_tensor[:, :, -len(target_cols):].cpu().numpy(),
            mask=val_mask.cpu().numpy(),
            y_scaler=y_scaler,
        )
        score = metrics["rmse"]

        print(
            f"n_epochs={n_epochs}, lr={lr}, tbptt_size={tbptt_size}, "
            f"weight_decay={wd}, smoothness_weight={smooth}, "
            f"rmse={metrics['rmse']:.4f}, mae={metrics['mae']:.4f}, "
            f"mse={metrics['mse']:.4f}, max_abs={metrics['max_abs']:.4f}"
        )

        if score < best_score:
            best_score = score
            best_params = {
                "n_epochs": n_epochs,
                "lr": lr,
                "tbptt_size": tbptt_size,
                "weight_decay": wd,
                "smoothness_weight": smooth,
                "n_neurons": n_neurons,
            }
            best_model = model

    if best_model is None or best_params is None:
        raise RuntimeError("Grid search failed to produce a best model.")

    return best_model, best_params, best_score



def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # =============================================================================
    # CONFIG
    # =============================================================================
    TS_COL = cfg["ts_col"]
    WINDOW_STEPS = cfg["window_steps"]
    TEST_PROFILE_LEN = cfg["test_profile_len"]
    VALIDATION_PROFILE_LEN = cfg["val_profile_len"]

    grid = cfg["grid"]
    grid_search = False

    params = cfg["default_params"]
    n_epochs = params["n_epochs"]
    lr = params["lr"]
    tbptt_size = params["tbptt_size"]
    weight_decay = params["weight_decay"]
    smoothness_weight = params["smoothness_weight"]
    best_n_neurons = params["n_neurons"]

    target_cols = cfg["target_cols"]
    temperature_cols = list(dict.fromkeys(cfg["temperature_cols"] + target_cols))
    excluded_cols = cfg["excluded_cols"]
    cooling_columns = cfg["cooling_columns"]

    drop_columns = [
                       "profile_id",
                       TS_COL,
                       "Modified",
                       "Inserted",
                       "AE PS POWER COUNTER",
                       "AE PS RUNNING",
                   ] + excluded_cols


    # =============================================================================
    # 1) LOAD DATA
    # =============================================================================
    dataset=1
    #split_NUMERIC_DIR = paths.DATASET_PATH / "split" / f"Dataset_{dataset}" / "numeric" / "AE_PORT.csv"
    split_NUMERIC_DIR = paths.DATASET_PATH / "split" / "temp" / "AE_PORT_with_extra_TEMP.csv"
    data = pd.read_csv(split_NUMERIC_DIR)
    # =============================================================================
    # 2) TIMESTAMP CLEANING
    # =============================================================================
    data , dt_s= ts_cols(data, TS_COL)
    # =============================================================================
    # 3) FEATURE PROCESSING
    # =============================================================================
    data = feature_processing(data)

    target_cols, temperature_cols, non_temperature_cols, input_cols = get_column_groups(
        data,
        target_cols=target_cols,
        temperature_cols=temperature_cols,
        drop_cols=drop_columns,
    )

    print("LEN TARGET COLS:", len(target_cols))
    print("TARGET COLS:", target_cols)
    print("LEN TEMPERATURE COLS:", len(temperature_cols))
    print("TEMPERATURE COLS:", temperature_cols)
    print("LEN NON-TEMPERATURE COLS:", len(non_temperature_cols))
    print("NON-TEMPERATURE COLS:", non_temperature_cols)
    print("LEN INPUT COLS:", len(input_cols))
    print("INPUT COLS:", input_cols)
    # =============================================================================
    # 5 SPLIT INTO TRAIN/ VALIDATION / TEST SET
    # =============================================================================
    data, train_profiles, val_profiles, test_profiles, profile_sizes = train_val_test_split_profiles(
        data,
        ts_col=TS_COL,
        window_steps=WINDOW_STEPS,
        val_profile_len=VALIDATION_PROFILE_LEN,
        test_profile_len=TEST_PROFILE_LEN,
        dt_s=dt_s,
    )
    # =============================================================================
    # 6) SCALING
    # =============================================================================
    data, x_scaler, y_scaler = scale_profile_data(
        data,
        train_profiles,
        input_cols,
        target_cols,
    )
    # =============================================================================
    # 7) TENSORISE
    # =============================================================================
    x_train, y_train, train_mask = tensorize_profiles(
        df=data,
        profiles=train_profiles,
        input_cols=input_cols,
        target_cols=target_cols,
        device=device
    )

    x_val, y_val, val_mask = tensorize_profiles(
        df=data,
        profiles=val_profiles,
        input_cols=input_cols,
        target_cols=target_cols,
        device=device
    )

    x_test, y_test, test_mask = tensorize_profiles(
        df=data,
        profiles=test_profiles,
        input_cols=input_cols,
        target_cols=target_cols,
        device=device
    )

    train_tensor = torch.cat([x_train, y_train], dim=2)
    val_tensor = torch.cat([x_val, y_val], dim=2)
    test_tensor = torch.cat([x_test, y_test], dim=2)


    # =============================================================================
    # 8) Build Model
    # =============================================================================
    model = build_model(
        dt_s=dt_s,
        input_cols=input_cols,
        target_cols=target_cols,
        temperature_cols=temperature_cols,
        device=device,
        cooling_columns=cooling_columns,
        n_neurons=best_n_neurons
    )
    # =============================================================================
    # 9) Train model
    # =============================================================================
    best_params = None

    if grid_search:
        model, best_params, best_score = run_grid_search(
            grid=grid,
            dt_s=dt_s,
            input_cols=input_cols,
            target_cols=target_cols,
            temperature_cols=temperature_cols,
            cooling_columns=cooling_columns,
            train_tensor=train_tensor,
            train_mask=train_mask,
            val_tensor=val_tensor,
            val_mask=val_mask,
            y_scaler=y_scaler,
            device=device,
        )

        print("Best params:", best_params)
        print(f"Best validation RMSE: {best_score:.4f}")

        best_n_neurons = best_params["n_neurons"]

        pred_c = predict_test_set(
            model=model,
            test_tensor=test_tensor,
            input_cols=input_cols,
            target_cols=target_cols,
            y_scaler=y_scaler,
        )

        metrics = evaluate_predictions(
            pred_c=pred_c,
            y_true_scaled=test_tensor[:, :, -len(target_cols):].cpu().numpy(),
            mask=test_mask.cpu().numpy(),
            y_scaler=y_scaler,
        )

        print_metrics(metrics)
    else:

        model = train_model(
            model=model,
            train_tensor=train_tensor,
            train_mask=train_mask,
            input_cols=input_cols,
            target_cols=target_cols,
            dt_s=dt_s,
            n_epochs=n_epochs,
            tbptt_size=tbptt_size,
            lr=lr,
            weight_decay=weight_decay,
            smoothness_weight=smoothness_weight,
        )
        pred_c = predict_test_set(
            model=model,
            test_tensor=test_tensor,
            input_cols=input_cols,
            target_cols=target_cols,
            y_scaler=y_scaler,
        )

        metrics = evaluate_predictions(
            pred_c=pred_c,
            y_true_scaled=test_tensor[:, :, -len(target_cols):].cpu().numpy(),
            mask=test_mask.cpu().numpy(),
            y_scaler=y_scaler,
        )
        print_metrics(metrics)

    # =============================================================================
    # 10) Evaluate model
    # =============================================================================
    eval_plot_from_predictions(
        pred_c=pred_c,
        test_mask=test_mask,
        data=data,
        test_profiles=test_profiles,
        target_cols=target_cols,
        y_scaler=y_scaler,
        ts_col=TS_COL,
    )
    # =============================================================================
    # 11) SAVE MODEL
    # =============================================================================
    save_model(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        input_cols=input_cols,
        target_cols=target_cols,
        temperature_cols=temperature_cols,
        dt_s=dt_s,
        n_neurons=best_n_neurons,
        cooling_columns=cooling_columns,
        path=paths.file_TNN_MODEL_cyl3,
        best_params=best_params if grid_search else {
            "n_epochs": n_epochs,
            "lr": lr,
            "tbptt_size": tbptt_size,
            "weight_decay": weight_decay,
            "smoothness_weight": smoothness_weight,
            "n_neurons": best_n_neurons,
        }
    )
if __name__ == "__main__":
    main()
