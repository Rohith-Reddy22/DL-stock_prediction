"""
run_phase2.py
-------------
Entry point for Phase 2: FinBERT Embedding Generation.

Loads the aligned dataset from Phase 1 to get the exact list of trading
dates we need embeddings for, then runs the embedding pipeline.

Usage:
    python run_phase2.py

Prerequisites:
    - Phase 1 must be complete (data/processed/aligned_dataset.csv exists)
    - data/raw/financial_news.csv must exist
    - ~440MB for FinBERT model download on first run (cached by HuggingFace)
    - Estimated runtime: 5-15 minutes on Apple Silicon M2 MPS
"""

import sys
import torch

sys.path.insert(0, ".")

import pandas as pd

from configs.data_config import DataConfig
from embeddings.embedding_pipeline import generate_embeddings
from utils.logger import get_logger

LOG_PATH = "logs/phase2.log"


def main() -> None:
    logger = get_logger(__name__, LOG_PATH)

    config = DataConfig(
        ticker="AAPL",
        start_date="2018-01-01",
        end_date="2020-06-04",
    )

    # ── Device detection ──────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info("=" * 60)
    logger.info("PHASE 2 — FinBERT Embedding Generation")
    logger.info("Device: %s", device)
    logger.info("=" * 60)

    # ── Load trading dates from Phase 1 output ────────────────────────────────
    aligned_path = config.aligned_data_path
    logger.info("Loading aligned dataset from %s ...", aligned_path)

    aligned_df = pd.read_csv(aligned_path, index_col="Date", parse_dates=True)
    aligned_df.index = aligned_df.index.normalize()

    # We need embeddings for T-1 (the day BEFORE each prediction date)
    # Phase 3 will handle the T-1 lookup; here we generate for all dates
    # in the aligned dataset so Phase 3 has maximum flexibility.
    trading_dates = sorted(aligned_df.index.strftime("%Y-%m-%d").tolist())

    logger.info("Trading dates to cover: %d", len(trading_dates))
    logger.info("Date range: %s → %s", trading_dates[0], trading_dates[-1])

    # ── Run embedding pipeline ────────────────────────────────────────────────
    embedding_index = generate_embeddings(
        news_csv_path="data/raw/financial_news.csv",
        trading_dates=trading_dates,
        start_date=config.start_date,
        end_date=config.end_date,
        output_dir="embeddings",
        device=device,
        log_path=LOG_PATH,
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    covered = [d for d in trading_dates if d in embedding_index]

    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print(f"  Device used        : {device}")
    print(f"  Trading days total : {len(trading_dates)}")
    print(f"  Embeddings created : {len(embedding_index)}")
    print(f"  Coverage           : {len(covered)/len(trading_dates)*100:.1f}%")
    print(f"  Saved to           : embeddings/")
    print(f"  Index file         : embeddings/embedding_index.json")
    print("=" * 60)

    # Verify one embedding loads correctly
    if embedding_index:
        sample_date = sorted(embedding_index.keys())[0]
        sample_path = embedding_index[sample_date]
        sample_emb = torch.load(sample_path, weights_only=True)
        print(f"\nSample check [{sample_date}]:")
        print(f"  Shape : {sample_emb.shape}")
        print(f"  Dtype : {sample_emb.dtype}")
        print(f"  Mean  : {sample_emb.mean():.4f}")
        print(f"  Std   : {sample_emb.std():.4f}")
        print(f"  NaNs  : {torch.isnan(sample_emb).sum().item()}")


if __name__ == "__main__":
    main()