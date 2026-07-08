"""
run_train.py
------------
Entry point for Phase 4: trains all 9 model configurations.

Usage:
    python run_train.py

Prerequisites:
    - Phase 1, 2, 3 must be complete
    - Estimated runtime: 10-30 minutes on Apple Silicon M2
"""

import sys
sys.path.insert(0, ".")

import torch

from configs.data_config import DataConfig
from datasets.dataloader import build_dataloaders
from training.experiment import run_all_experiments
from utils.logger import get_logger
from utils.seed import set_seed

LOG_PATH = "logs/training.log"


def main() -> None:
    logger = get_logger(__name__, LOG_PATH)

    config = DataConfig(
        ticker="AAPL",
        start_date="2018-01-01",
        end_date="2020-06-04",
    )

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info("=" * 60)
    logger.info("PHASE 4 — Training All Experiments")
    logger.info("Device: %s", device)
    logger.info("Experiments: 3 variants × 3 hidden sizes = 9 runs")
    logger.info("=" * 60)

    set_seed(42)

    # ── DataLoaders ───────────────────────────────────────────────────────────
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

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = torch.load(
        config.class_weights_path,
        map_location="cpu",
        weights_only=True,
    )
    logger.info("Class weights loaded: %s", class_weights.tolist())

    # ── Run all experiments ───────────────────────────────────────────────────
    results_df = run_all_experiments(
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        results_dir="evaluation",
        checkpoint_dir="checkpoints",
        log_path=LOG_PATH,
        seed=42,
    )

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 4 COMPLETE — Ablation Results Summary")
    print("=" * 65)
    display_cols = ["variant", "hidden_size", "best_epoch", "best_val_loss", "best_val_acc", "n_params"]
    print(results_df[display_cols].to_string(index=False))
    print("=" * 65)
    print(f"\nFull results saved to: evaluation/ablation_results.csv")
    print(f"Checkpoints saved to:  checkpoints/")
    print(f"\nNext step: python run_eval.py")


if __name__ == "__main__":
    main()