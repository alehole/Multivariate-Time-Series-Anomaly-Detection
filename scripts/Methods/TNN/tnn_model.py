from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter as TorchParam
from torch import Tensor

def build_model(
    dt_s: float,
    input_cols: list[str],
    target_cols: list[str],
    temperature_cols: list[str],
    device: torch.device,
    cooling_columns: list[str] | None = None,
    n_neurons: int = 16,
) -> nn.Module:

    # T_k+1 = Tk+Δt⋅gain⋅(heat flow+internal losses+cooling+bias
    extra_temp_cols = [
        c for c in temperature_cols
        if c not in target_cols and c in input_cols
    ]

    class DiffEqLayer(nn.Module):
        def __init__(self, cell_cls):
            super().__init__()
            self.cell = cell_cls()

        def forward(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
            """
           Time integration layer for the thermal neural network.

           Dimensions
           ----------
           B : batch size (number of profiles)
           T : number of timesteps
           F_in : number of input features
           S : number of thermal states (targets)

           Parameters
           ----------
           x : Tensor
               Input sequence of shape (B, T, F_in)

           state : Tensor
               Initial thermal state of shape (B, S)

           Returns
           -------
           out_seq : Tensor
               Predicted temperature sequence of shape (B, T, S)

           state : Tensor
               Final thermal state after the last timestep, shape (B, S)
           """
            outs = []
            for x_t in x.unbind(dim=1):
                out, state = self.cell(x_t, state)
                outs.append(out)
            out_seq = torch.stack(outs, dim=1)
            return out_seq, state

    class TNNCell(nn.Module):
        def __init__(self):
            super().__init__()

            self.sample_time = dt_s
            self.output_size = len(target_cols)

            # Learnable constant heat-source / offset term for each target node
            self.bias_loss = nn.Parameter(torch.zeros(self.output_size, device=device))

            # Learned positive gain term, acting like an inverse thermal capacitance
            self.caps = TorchParam(torch.empty(self.output_size, device=device))
            #nn.init.normal_(self.caps, mean=-9.2, std=0.5)
            nn.init.normal_(self.caps, mean=-7.5, std=0.4)

            # Number of thermal nodes (temperature signals)
            n_temps = len(temperature_cols)

            # Number of unique pairwise thermal connections in a fully connected graph
            n_conds = int(0.5 * n_temps * (n_temps - 1))

            print("n_temps", n_temps) # 13
            print("n_conds", n_conds) # 78

            in_dim = len(input_cols) + self.output_size
            # Fully connected thermal graph:
            # at each timestep, the model estimates pairwise thermal conductances G_ij
            self.conductance_net = nn.Sequential(
                nn.Linear(in_dim, n_conds),
                #nn.Sigmoid(),
                nn.Softplus()
            )

            # Learned internal heat-generation term (e.g. copper/iron losses)
            self.ploss = nn.Sequential(
                nn.Linear(in_dim, n_neurons),
                nn.Tanh(),
                nn.Linear(n_neurons, self.output_size),
            )

            # Adjacency-style index map from flattened pairwise conductances
            # to a node-to-node conductance matrix
            adj_mat = np.zeros((n_temps, n_temps), dtype=int)
            triu_idx = np.triu_indices(n_temps, 1)
            adj_mat[triu_idx] = np.arange(len(triu_idx[0]))
            adj_mat = adj_mat + adj_mat.T
            self.adj_mat = torch.from_numpy(adj_mat[: self.output_size, :]).long().to(device)

            # indices of measured temperature inputs among input_cols
            self.temp_idcs = [i for i, name in enumerate(input_cols) if name in extra_temp_cols]

            # Convection
            # Learnable cooling strength toward a sink temperature
            self.k_sink = nn.Parameter(torch.zeros(self.output_size, device=device))

            # Store all matching cooling sensor indices
            self.sink_idcs: list[int] = []
            if cooling_columns is not None:
                self.sink_idcs = [
                    input_cols.index(name)
                    for name in cooling_columns
                    if name in input_cols
                ]

            print("Cooling columns used:",
                  [input_cols[i] for i in self.sink_idcs] if self.sink_idcs else "None")

        def forward(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
            """
            B    : batch size
            F_in : number of input features
            S    : number of target thermal states

            x     : (B, F_in)
            state : (B, S)
            """
            prev_out = state  # Current thermal state

            # Thermal nodes = predicted target temperatures + measured temperature inputs
            if self.temp_idcs:
                temps = torch.cat([prev_out, x[:, self.temp_idcs]], dim=1)
            else:
                temps = prev_out

            # Input to learned thermal submodels
            inp = torch.cat([x, prev_out], dim=1)

            # Learned positive pairwise thermal conductances
            conducts = torch.abs(self.conductance_net(inp))   # (B, n_conds)

            # Learned internal heat generation / thermal loss term
            power_loss = self.ploss(inp)                      # (B, S)

            # Conductive heat exchange between thermal nodes
            node_cond = conducts[:, self.adj_mat]             # (B, S, n_temps)
            temp_diff = temps.unsqueeze(1) - prev_out.unsqueeze(-1)  # (B, S, n_temps)
            heat_flow = (node_cond * temp_diff).sum(dim=-1)   # (B, S)

            # Learnable constant heat-source / offset term for each target node
            bias = self.bias_loss.unsqueeze(0)                # (1, S)

            # Average cooling sink temperature from all cooling sensors
            if self.sink_idcs:
                sink = x[:, self.sink_idcs].mean(dim=1, keepdim=True)   # (B, 1)
            else:
                sink = torch.zeros((prev_out.shape[0], 1), device=prev_out.device)

            # Positive cooling coefficient
            k = torch.nn.functional.softplus(self.k_sink).unsqueeze(0)  # (1, S)

            # # Convection-like cooling toward sink temperature
            cooling = k * (sink - prev_out)                              # (B, S)

            # Learned positive gain term, inverse thermal capacitance.
            gain = torch.exp(self.caps).unsqueeze(0)  # (1, S)

            # Forward-Euler thermal state update
            next_state = prev_out + self.sample_time * gain * (
                heat_flow + power_loss + cooling + bias
            )

            return next_state, next_state

    class TNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = DiffEqLayer(TNNCell)

        def forward(self, x: Tensor, state0: Tensor | None = None) -> tuple[Tensor, Tensor]:
            """
            B    : batch size
            T    : number of timesteps
            F_in : number of input features
            S    : number of target thermal states

            Parameters
            ----------
            x : Tensor
                Input sequence of shape (B, T, F_in)

            state0 : Tensor | None
                Initial thermal state of shape (B, S).
                If None, zero initialization is used.

            Returns
            -------
            y_hat : Tensor
                Predicted temperature sequence of shape (B, T, S)

            final_state : Tensor
                Final thermal state after the last timestep, shape (B, S)
            """
            B = x.shape[0]
            if state0 is None:
                state0 = torch.zeros(B, len(target_cols), device=x.device)
            y_hat, final_state = self.layer(x, state0)
            return y_hat, final_state

    return TNNModel().to(device)