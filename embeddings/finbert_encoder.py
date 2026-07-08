"""
embeddings/finbert_encoder.py
------------------------------
Loads ProsusAI/finbert once and encodes text into 768-dim embeddings using
attention-mask-aware mean pooling over the last hidden state.

Architecture decisions (per spec):
- Mean pooling over last_hidden_state, weighted by attention mask.
  This correctly ignores padding tokens when averaging.
- CLS token ([0]) is explicitly NOT used — mean pooling over all non-padding
  tokens captures richer sentence-level semantics for longer concatenated text.
- FinBERT is loaded in eval() mode with torch.no_grad() throughout.
- Parameters are frozen — no gradients are computed or stored.
- MPS is used on Apple Silicon. A NaN safety check falls back to CPU
  for any batch that produces NaN values (rare MPS precision issue).

Token limit:
- FinBERT max sequence length is 512 tokens.
- We pass truncation=True and max_length=512 to the tokenizer.
- Long concatenated headlines are truncated at the token level,
  preserving the most semantically important leading content.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from utils.logger import get_logger

FINBERT_MODEL_NAME = "ProsusAI/finbert"
EMBEDDING_DIM = 768


class FinBERTEncoder:
    """
    Wrapper around ProsusAI/finbert for pre-computing text embeddings.

    Usage:
        encoder = FinBERTEncoder(device)
        embedding = encoder.encode("Apple beats earnings expectations.")
        # embedding.shape == (768,)
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        log_path: str = "logs/phase2.log",
    ) -> None:
        """
        Load FinBERT tokenizer and model onto the specified device.

        Args:
            device:   torch.device to run inference on.
                      Defaults to MPS → CUDA → CPU auto-detection.
            log_path: Path for log output.
        """
        self.logger = get_logger(__name__, log_path)

        # ── Device selection ──────────────────────────────────────────────────
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

        self.device = device
        self.logger.info("Loading FinBERT on device: %s", self.device)

        # ── Load tokenizer and model ──────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        self.model = AutoModel.from_pretrained(FINBERT_MODEL_NAME)

        # Freeze all parameters — FinBERT is never fine-tuned
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model.to(self.device)

        self.logger.info(
            "FinBERT loaded. Parameters: %d (all frozen).",
            sum(p.numel() for p in self.model.parameters()),
        )

    @staticmethod
    def _mean_pool(
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Attention-mask-aware mean pooling over the last hidden state.

        Padding tokens (attention_mask == 0) are excluded from the average.

        Args:
            last_hidden_state: (batch, seq_len, hidden_size)
            attention_mask:    (batch, seq_len)  — 1 for real tokens, 0 for padding

        Returns:
            (batch, hidden_size) mean-pooled embeddings.
        """
        # Expand mask to match hidden state dimensions
        mask_expanded = attention_mask.unsqueeze(-1).expand(
            last_hidden_state.size()
        ).float()

        # Sum embeddings of real tokens only
        sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)

        # Count real tokens per sample (clamp to avoid div-by-zero)
        token_counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)

        return sum_embeddings / token_counts

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a text string into a 768-dim embedding vector.

        Args:
            text: Concatenated headline string for one trading day.

        Returns:
            CPU FloatTensor of shape (768,).
            Always returned on CPU regardless of inference device,
            so it can be safely saved with torch.save().
        """
        # ── Tokenise ──────────────────────────────────────────────────────────
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # ── Forward pass ─────────────────────────────────────────────────────
        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = self._mean_pool(
            outputs.last_hidden_state,
            inputs["attention_mask"],
        ).squeeze(0)  # (768,)

        # ── NaN safety check (rare MPS precision issue) ───────────────────────
        if torch.isnan(embedding).any():
            self.logger.warning(
                "NaN detected in embedding on %s — retrying on CPU.", self.device
            )
            cpu_inputs = {k: v.cpu() for k, v in inputs.items()}
            self.model.cpu()
            with torch.no_grad():
                cpu_outputs = self.model(**cpu_inputs)
            embedding = self._mean_pool(
                cpu_outputs.last_hidden_state,
                cpu_inputs["attention_mask"],
            ).squeeze(0)
            self.model.to(self.device)

        return embedding.cpu()