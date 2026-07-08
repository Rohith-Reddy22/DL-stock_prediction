"""
models/lstm_branch.py
----------------------
2-layer LSTM branch for numerical time-series input.

Input:  (batch, sequence_length, num_features) = (B, 30, 13)
Output: (batch, hidden_size)  — final hidden state of last LSTM layer

Architecture (per spec):
    2-layer LSTM
    hidden_size : configurable (64 / 128 / 256)
    dropout     : 0.2 between layers
    batch_first : True
"""

import torch
import torch.nn as nn


class LSTMBranch(nn.Module):
    """
    2-layer LSTM that encodes a 30-day price/indicator sequence.

    Args:
        input_size:  Number of features per timestep (13).
        hidden_size: LSTM hidden dimension. One of {64, 128, 256}.
        num_layers:  Number of stacked LSTM layers (default 2, per spec).
        dropout:     Dropout between LSTM layers (default 0.2, per spec).
    """

    def __init__(
        self,
        input_size: int = 13,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape (batch, seq_len, input_size)

        Returns:
            FloatTensor of shape (batch, hidden_size)
            — the final hidden state from the last LSTM layer.
        """
        # lstm output: (batch, seq_len, hidden_size)
        # h_n shape:   (num_layers, batch, hidden_size)
        _, (h_n, _) = self.lstm(x)

        # Take the hidden state from the last layer
        return h_n[-1]  # (batch, hidden_size)