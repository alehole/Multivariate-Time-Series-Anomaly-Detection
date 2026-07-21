# ============================================================
# LSTM / GRU BLACK-BOX BASELINE
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


class RNNBaseline(nn.Module):
    """
    Black-box sequence model for multivariate time-series prediction.

    Inputs:
        x: (B, T, F_in)

    Outputs:
        y_hat: (B, T, F_out)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        model_type: str = "GRU",   # "GRU" or "LSTM"
    ):
        super().__init__()

        self.model_type = model_type.upper()

        if self.model_type == "LSTM":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        elif self.model_type == "GRU":
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        else:
            raise ValueError("model_type must be either 'GRU' or 'LSTM'")

        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.rnn(x)
        y_hat = self.head(out)
        return y_hat


def masked_mse_loss(y_hat, y_true, mask):
    """
    y_hat:  (B, T, F)
    y_true: (B, T, F)
    mask:   (B, T)
    """

    mask = mask.unsqueeze(-1)  # (B, T, 1)

    loss = (y_hat - y_true) ** 2
    loss = loss * mask

    return loss.sum() / mask.sum().clamp(min=1.0)

def train_sequence_model(
    model,
    x_train,
    y_train,
    mask_train,
    x_val=None,
    y_val=None,
    mask_val=None,
    *,
    n_epochs=200,
    lr=1e-3,
    weight_decay=1e-5,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, n_epochs + 1):
        model.train()

        optimizer.zero_grad()

        y_hat = model(x_train)
        loss = masked_mse_loss(y_hat, y_train, mask_train)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss = loss.item()
        history["train_loss"].append(train_loss)

        if x_val is not None:
            model.eval()
            with torch.no_grad():
                y_val_hat = model(x_val)
                val_loss = masked_mse_loss(y_val_hat, y_val, mask_val).item()

            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

        else:
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.6f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history