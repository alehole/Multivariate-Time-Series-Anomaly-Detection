# ============================================================
# LSTM / GRU
# ============================================================
import torch.nn as nn
from torch import Tensor

class RNNBaseline(nn.Module):
    """
      Recurrent black-box model for multivariate time-series prediction.

      The model is unidirectional, so the prediction at time step ``t`` only
      depends on the current and preceding input samples.

      Parameters
      ----------
      input_size:
          Number of input variables.
      output_size:
          Number of predicted variables.
      hidden_size:
          Number of features in the recurrent hidden state.
      num_layers:
          Number of stacked recurrent layers.
      dropout:
          Dropout applied between recurrent layers. PyTorch only applies this
          when ``num_layers > 1``.
      model_type:
          Either ``"LSTM"`` or ``"GRU"``.

      Input shape
      -----------
      (B, T, F_in)

      Output shape
      ------------
      (B, T, F_out)
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
            recurrent_class = nn.LSTM
        elif self.model_type == "GRU":
            recurrent_class = nn.GRU
        else:
            raise ValueError("model_type must be either 'LSTM' or 'GRU'.")

        self.rnn = recurrent_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.rnn(x)
        y_hat = self.head(out)
        return y_hat