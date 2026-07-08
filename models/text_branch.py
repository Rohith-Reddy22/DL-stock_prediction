"""
models/text_branch.py
----------------------
Text branch that receives a pre-computed frozen FinBERT embedding (768-dim)
and passes it through a lightweight projection layer.

Design decision:
- The embedding is already 768-dim and semantically rich. A single linear
  projection normalises the scale before fusion without destroying structure.
- No activation after projection — the fusion head applies ReLU downstream.
- FinBERT weights are NEVER loaded here — embeddings are pre-computed in
  Phase 2 and loaded as tensors. This module only projects them.

Input:  (batch, 768)
Output: (batch, 768)   — same dim, but linearly projected and normalised
"""

import torch
import torch.nn as nn


class TextBranch(nn.Module):
    """
    Lightweight projection of pre-computed FinBERT embeddings.

    For the TextOnly ablation variant the output feeds directly into
    the classification head. For the Fusion variant it is concatenated
    with the LSTM output before the head.

    Args:
        embedding_dim: Input embedding dimension (768, fixed by FinBERT).
        output_dim:    Output dimension (default 768 — preserves FinBERT dim).
        dropout:       Dropout rate applied before projection (default 0.1).
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        output_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, output_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embedding: FloatTensor of shape (batch, 768)

        Returns:
            FloatTensor of shape (batch, output_dim)
        """
        return self.projection(embedding)