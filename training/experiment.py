"""
training/experiment.py
-----------------------
Runs all 9 experiments (3 variants × 3 hidden sizes) and saves a full
ablation results table to evaluation/ablation_results.csv.

Experiment grid:
    variants     : ["price_only", "text_only", "fusion"]
    hidden_sizes : [64, 128, 256]

    text_only variant ignores hidden_size (no LSTM) — still run 3 times
    for a fair comparison table, but results will be identical across
    hidden sizes (logged with a note).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader

from models.fusion_model import FusionModel, VariantType
from training.trainer import TrainerConfig, train
from utils.logger import get_logger
from utils.seed import set_seed


VARIANTS: List[VariantType] = ["price_only", "text_only", "fusion"]
HIDDEN_SIZES: List[int] = [64, 128, 256]


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_all_experiments(
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    results_dir: str = "evaluation",
    checkpoint_dir: str = "checkpoints",
    log_path: str = "logs/training.log",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Train all 9 model configurations and return a results DataFrame.

    Args:
        train_loader:   Training DataLoader.
        val_loader:     Validation DataLoader.
        class_weights:  FloatTensor [w0, w1].
        results_dir:    Where to save ablation_results.csv.
        checkpoint_dir: Where to save per-run checkpoints.
        log_path:       Log file path.
        seed:           Random seed for reproducibility.

    Returns:
        DataFrame with columns:
        [variant, hidden_size, best_epoch, best_val_loss, best_val_acc,
         n_params, checkpoint_path]
    """
    logger = get_logger(__name__, log_path)
    device = get_device()

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    rows = []

    for variant in VARIANTS:
        for hidden_size in HIDDEN_SIZES:

            # text_only doesn't use hidden_size — skip duplicates after first
            if variant == "text_only" and hidden_size != 64:
                logger.info(
                    "Skipping %s h%d — text_only variant is hidden_size-independent.",
                    variant, hidden_size,
                )
                # Still add a row pointing to the h64 checkpoint
                rows.append({
                    "variant": variant,
                    "hidden_size": hidden_size,
                    "best_epoch": rows[-1]["best_epoch"] if rows else None,
                    "best_val_loss": rows[-1]["best_val_loss"] if rows else None,
                    "best_val_acc": rows[-1]["best_val_acc"] if rows else None,
                    "n_params": rows[-1]["n_params"] if rows else None,
                    "checkpoint_path": rows[-1]["checkpoint_path"] if rows else None,
                    "note": "same as text_only_h64",
                })
                continue

            run_name = f"{variant}_h{hidden_size}"
            logger.info("=" * 55)
            logger.info("EXPERIMENT: %s", run_name)
            logger.info("=" * 55)

            set_seed(seed)

            model = FusionModel(
                variant=variant,
                hidden_size=hidden_size,
            )

            trainer_config = TrainerConfig(
                learning_rate=0.001,
                lr_factor=0.5,
                lr_patience=5,
                lr_min=1e-6,
                early_stop_patience=20,
                max_epochs=150,
                device=str(device),
                checkpoint_dir=checkpoint_dir,
                log_path=log_path,
                run_name=run_name,
            )

            trained_model, history, ckpt_path = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                config=trainer_config,
            )

            # Best epoch metrics
            best = min(history, key=lambda m: m.val_loss)

            rows.append({
                "variant": variant,
                "hidden_size": hidden_size,
                "best_epoch": best.epoch,
                "best_val_loss": round(best.val_loss, 4),
                "best_val_acc": round(best.val_acc, 4),
                "n_params": trained_model.count_parameters(),
                "checkpoint_path": ckpt_path,
                "note": "",
            })

            logger.info(
                "DONE %s | best_epoch=%d | val_loss=%.4f | val_acc=%.4f | params=%d",
                run_name, best.epoch, best.val_loss, best.val_acc,
                trained_model.count_parameters(),
            )

    results_df = pd.DataFrame(rows)

    # Save results table
    results_path = Path(results_dir) / "ablation_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info("Ablation results saved to %s", results_path)

    # Also save as JSON for easy loading in Phase 5
    json_path = Path(results_dir) / "ablation_results.json"
    results_df.to_json(json_path, orient="records", indent=2)

    return results_df