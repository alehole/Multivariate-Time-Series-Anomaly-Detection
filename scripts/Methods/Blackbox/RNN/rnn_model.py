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



