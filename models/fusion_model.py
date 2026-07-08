"""
models/fusion_model.py
-----------------------
Late fusion multimodal model supporting three variants:

    "fusion"     : LSTM(hidden_size) + FinBERT(768) → concat → head
    "price_only" : LSTM(hidden_size) → head  (ablation: no text)
    "text_only"  : FinBERT(768) → head        (ablation: no price)

Fusion architecture (per spec):
    V_fusion = concat([V_lstm, V_finbert])   dim = hidden_size + 768
    → Linear(fusion_dim, 128)
    → ReLU
    → Dropout(0.3)
    → Linear(128, 2)

The classification head input dimension changes per variant:
    fusion     : hidden_size + 768
    price_only : hidden_size
    text_only  : 768
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from models.lstm_branch import LSTMBranch
from models.text_branch import TextBranch

VariantType = Literal["fusion", "price_only", "text_only"]


class FusionModel(nn.Module):
    """
    Configurable multimodal stock direction classifier.

    Args:
        variant:     One of "fusion", "price_only", "text_only".
        hidden_size: LSTM hidden dimension. One of {64, 128, 256}.
        input_size:  Number of price/indicator features (13).
        emb_dim:     FinBERT embedding dimension (768).
        head_hidden: Hidden units in classification head (128, per spec).
        dropout:     Dropout in classification head (0.3, per spec).
        lstm_dropout: Dropout between LSTM layers (0.2, per spec).
    """

    def __init__(
        self,
        variant: VariantType = "fusion",
        hidden_size: int = 64,
        input_size: int = 13,
        emb_dim: int = 768,
        head_hidden: int = 128,
        dropout: float = 0.3,
        lstm_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.variant = variant
        self.hidden_size = hidden_size

        # ── Branches (only instantiate what's needed) ─────────────────────────
        if variant in ("fusion", "price_only"):
            self.lstm_branch = LSTMBranch(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=lstm_dropout,
            )
        else:
            self.lstm_branch = None

        if variant in ("fusion", "text_only"):
            self.text_branch = TextBranch(
                embedding_dim=emb_dim,
                output_dim=emb_dim,
            )
        else:
            self.text_branch = None

        # ── Compute fusion input dimension ────────────────────────────────────
        if variant == "fusion":
            head_input_dim = hidden_size + emb_dim   # e.g. 64 + 768 = 832
        elif variant == "price_only":
            head_input_dim = hidden_size              # e.g. 64
        else:  # text_only
            head_input_dim = emb_dim                  # 768

        # ── Classification head (per spec) ────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(head_input_dim, head_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 2),
        )

    def forward(
        self,
        price_seq: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            price_seq:  FloatTensor (batch, seq_len, input_size) = (B, 30, 13)
            embedding:  FloatTensor (batch, 768)

        Returns:
            FloatTensor (batch, 2) — raw logits for CrossEntropyLoss.
        """
        if self.variant == "fusion":
            lstm_out = self.lstm_branch(price_seq)      # (B, hidden_size)
            text_out = self.text_branch(embedding)      # (B, 768)
            fused = torch.cat([lstm_out, text_out], dim=1)  # (B, hidden_size+768)
            return self.classifier(fused)

        elif self.variant == "price_only":
            lstm_out = self.lstm_branch(price_seq)      # (B, hidden_size)
            return self.classifier(lstm_out)

        else:  # text_only
            text_out = self.text_branch(embedding)      # (B, 768)
            return self.classifier(text_out)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)