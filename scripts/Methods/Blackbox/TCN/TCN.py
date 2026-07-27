import torch.nn as nn
import torch.nn.functional as F
#https://medium.com/@sahin.samia/mastering-the-basics-of-torch-nn-a-comprehensive-guide-to-pytorchs-neural-network-module-9f2d704e8c7f
class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout=0.1,
    ):
        super().__init__()

        # left-padding amount for causal convolution
        self.pad = (kernel_size - 1) * dilation

        # NOTE: no padding arg here — we pad manually on the left in forward
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        # F.pad(x, (left, right)) pads the last dim; (self.pad, 0) = left only
        out = self.conv1(F.pad(x, (self.pad, 0)))
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(F.pad(out, (self.pad, 0)))
        out = self.relu(out)
        out = self.dropout(out)

        residual = x if self.downsample is None else self.downsample(x)

        return self.relu(out + residual)

class TCNBaseline(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        channels=(32, 64, 64),
        kernel_size=3,
        dropout=0.1,
    ):
        super().__init__()

        layers = []

        in_ch = input_size

        for i, out_ch in enumerate(channels):

            dilation = 2 ** i

            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

            in_ch = out_ch

        self.network = nn.Sequential(*layers)

        self.head = nn.Conv1d(
            in_ch,
            output_size,
            kernel_size=1,
        )

    def forward(self, x):

        # (B,T,F) -> (B,F,T)
        x = x.transpose(1, 2)

        x = self.network(x)

        x = self.head(x)

        # (B,F,T) -> (B,T,F)
        x = x.transpose(1, 2)

        return x