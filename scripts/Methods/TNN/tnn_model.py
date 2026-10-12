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
    n_neurons: int = 16,
) -> nn.Module:

    # T_k+1 = Tk+Δt⋅gain⋅(heat flow+internal losses

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

            # --------------------------------------------------------
            # Inverse thermal capacitance
            # --------------------------------------------------------
            self.caps = TorchParam(
                torch.empty(
                    self.output_size,
                    device=device
                )
            )

            # initialization
            nn.init.normal_(self.caps, mean=-7.5, std=0.4)

            # --------------------------------------------------------
            # Thermal graph / Number of thermal nodes
            # --------------------------------------------------------
            n_temps = len(temperature_cols)              # number of temperatures
            n_conds = int(0.5 * n_temps * (n_temps - 1)) # number of thermal conductances
            print("n_temps", n_temps)
            print("n_conds", n_conds)
            in_dim = len(input_cols) + self.output_size

            # --------------------------------------------------------
            # Conductance network
            # --------------------------------------------------------
            self.conductance_net = nn.Sequential(
                nn.Linear(in_dim, n_conds),
                #nn.Sigmoid(),
                nn.Softplus()
            )

            # --------------------------------------------------------
            # Internal heat-generation network
            # --------------------------------------------------------
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

            # --------------------------------------------------------
            # Measured temperature inputs
            # --------------------------------------------------------
            extra_temp_cols = [
                c for c in temperature_cols
                if c not in target_cols and c in input_cols
            ]
            self.temp_idcs = [i for i, name in enumerate(input_cols) if name in extra_temp_cols]

        def forward(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
            """
            B    : batch size
            F_in : number of input features
            S    : number of target thermal states

            x     : (B, F_in)
            state : (B, S)
            """
            prev_out = state  # Current thermal state

            # --------------------------------------------------------
            # Thermal node temperatures
            # --------------------------------------------------------
            if self.temp_idcs:
                temps = torch.cat([prev_out, x[:, self.temp_idcs]], dim=1)
            else:
                temps = prev_out

            # --------------------------------------------------------
            # Input to the learned sub-networks
            # --------------------------------------------------------
            inp = torch.cat([x, prev_out], dim=1)

            # --------------------------------------------------------
            # Positive thermal conductances
            # --------------------------------------------------------
            conducts = torch.abs(self.conductance_net(inp))   # (B, n_conds)

            # --------------------------------------------------------
            # Positive internal heat generation
            # --------------------------------------------------------
            #power_loss= torch.abs(self.ploss(inp))        # Original formulation
            power_loss = self.ploss(inp)                 # (B, S)

            # --------------------------------------------------------
            # Conductive heat transfer
            # --------------------------------------------------------
            node_cond = conducts[:, self.adj_mat]             # (B, S, n_temps)
            temp_diff = temps.unsqueeze(1) - prev_out.unsqueeze(-1)  # (B, S, n_temps)
            heat_flow = (node_cond * temp_diff).sum(dim=-1)   # (B, S)

            # Learned positive gain term, inverse thermal capacitance.
            gain = torch.exp(self.caps).unsqueeze(0)  # (1, S)

            # Forward-Euler thermal state update
            next_state = prev_out + self.sample_time * gain * (heat_flow + power_loss)

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