"""
run_phase3.py
-------------
Entry point for Phase 3: Custom PyTorch Dataset validation.

Builds all three DataLoaders and runs shape/type/value checks
on a sample batch to verify correctness before Phase 4 training.

Usage:
    python run_phase3.py
"""

import sys
sys.path.insert(0, ".")

import torch

from configs.data_config import DataConfig
from datasets.dataloader import build_dataloaders
from utils.logger import get_logger

LOG_PATH = "logs/phase3.log"


def main() -> None:
    logger = get_logger(__name__, LOG_PATH)
    config = DataConfig(
        ticker="AAPL",
        start_date="2018-01-01",
        end_date="2020-06-04",
    )

    logger.info("=" * 60)
    logger.info("PHASE 3 — Dataset & DataLoader Validation")
    logger.info("=" * 60)

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        train_path="data/train_scaled.csv",
        val_path="data/val_scaled.csv",
        test_path="data/test_scaled.csv",
        full_data_path="data/processed/full_scaled.csv",
        embedding_index_path="embeddings/embedding_index.json",
        feature_cols=config.feature_cols,
        batch_size=32,
        sequence_length=config.sequence_length,
        log_path=LOG_PATH,
    )

    # ── Shape validation ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 3 — Shape & Type Validation")
    print("=" * 60)

    for split_name, loader in [
        ("TRAIN", train_loader),
        ("VAL",   val_loader),
        ("TEST",  test_loader),
    ]:
        price_seq, embedding, label = next(iter(loader))

        print(f"\n{split_name} split:")
        print(f"  price_sequence : {price_seq.shape}  dtype={price_seq.dtype}")
        print(f"  embedding      : {embedding.shape}  dtype={embedding.dtype}")
        print(f"  label          : {label.shape}  dtype={label.dtype}")
        print(f"  label values   : {label.tolist()[:8]} ...")
        print(f"  price min/max  : [{price_seq.min():.4f}, {price_seq.max():.4f}]")
        print(f"  emb mean/std   : {embedding.mean():.4f} / {embedding.std():.4f}")

        # ── Assertions ────────────────────────────────────────────────────────
        B = price_seq.shape[0]
        assert price_seq.shape == (B, 30, 13), \
            f"Expected ({B}, 30, 13), got {price_seq.shape}"
        assert embedding.shape == (B, 768), \
            f"Expected ({B}, 768), got {embedding.shape}"
        assert label.shape == (B,), \
            f"Expected ({B},), got {label.shape}"
        assert price_seq.dtype == torch.float32, "price_seq must be float32"
        assert embedding.dtype == torch.float32, "embedding must be float32"
        assert label.dtype == torch.long, "label must be long (int64)"
        assert set(label.tolist()).issubset({0, 1}), "labels must be 0 or 1"
        assert not torch.isnan(price_seq).any(), "NaN in price_sequence"
        assert not torch.isnan(embedding).any(), "NaN in embedding"

        print(f"  ✓ All assertions passed")

    # ── Dataset summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"  Train samples  : {len(train_loader.dataset)}")
    print(f"  Val samples    : {len(val_loader.dataset)}")
    print(f"  Test samples   : {len(test_loader.dataset)}")
    print(f"  Batch size     : 32")
    print(f"  Train batches  : {len(train_loader)}")
    print(f"  Val batches    : {len(val_loader)}")
    print(f"  Test batches   : {len(test_loader)}")
    print(f"  Input shapes   : price=(B,30,13) | emb=(B,768) | label=(B,)")
    print("=" * 60)
    print("\nPhase 3 complete. Ready for Phase 4 training.")


if __name__ == "__main__":
    main()