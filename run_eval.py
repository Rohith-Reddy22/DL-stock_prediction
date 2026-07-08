"""
run_eval.py
-----------
Phase 5: Full evaluation of all trained models on the held-out test set.

Loads each checkpoint, runs inference, computes all metrics, generates
all plots, and saves a final results table.

Usage:
    python run_eval.py
"""

import json
import sys
sys.path.insert(0, ".")

from pathlib import Path

import pandas as pd
import torch

from configs.data_config import DataConfig
from datasets.dataloader import build_dataloaders
from evaluation.evaluator import evaluate_on_test
from evaluation.plots import (
    plot_ablation_heatmap,
    plot_confusion_matrix,
    plot_hidden_size_comparison,
    plot_loss_curves,
    plot_roc_auc_comparison,
)
from models.fusion_model import FusionModel
from training.experiment import HIDDEN_SIZES, VARIANTS
from training.trainer import EpochMetrics, TrainerConfig, train
from utils.logger import get_logger
from utils.seed import set_seed

LOG_PATH = "logs/evaluation.log"
PLOT_DIR = "evaluation/plots"
RESULTS_DIR = "evaluation"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_history_from_log(run_name: str) -> list:
    """
    Reconstruct training history by re-reading the training log.
    Returns list of EpochMetrics for loss curve plotting.
    """
    import re
    history = []
    log_path = Path("logs/training.log")
    if not log_path.exists():
        return history

    pattern = re.compile(
        rf"\[{re.escape(run_name)}\] Epoch\s+(\d+) \| "
        r"train_loss=([\d.]+) \| val_loss=([\d.]+) \| "
        r"val_acc=([\d.]+) \| lr=([\de.+-]+)"
    )

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                history.append(EpochMetrics(
                    epoch=int(m.group(1)),
                    train_loss=float(m.group(2)),
                    val_loss=float(m.group(3)),
                    val_acc=float(m.group(4)),
                    lr=float(m.group(5)),
                ))
    return history


def main() -> None:
    logger = get_logger(__name__, LOG_PATH)
    device = get_device()
    set_seed(42)

    config = DataConfig(
        ticker="AAPL",
        start_date="2018-01-01",
        end_date="2020-06-04",
    )

    logger.info("=" * 60)
    logger.info("PHASE 5 — Evaluation & Visualisation")
    logger.info("Device: %s", device)
    logger.info("=" * 60)

    Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    _, _, test_loader = build_dataloaders(
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

    # ── Load Phase 4 ablation table ───────────────────────────────────────────
    ablation_path = Path(RESULTS_DIR) / "ablation_results.csv"
    ablation_df = pd.read_csv(ablation_path)

    # ── Evaluate each trained model ───────────────────────────────────────────
    all_test_results = []

    for variant in VARIANTS:
        for hidden_size in HIDDEN_SIZES:

            run_name = f"{variant}_h{hidden_size}"

            # text_only h128/h256 share the h64 checkpoint
            ckpt_run = (
                f"{variant}_h64"
                if variant == "text_only" and hidden_size != 64
                else run_name
            )
            ckpt_path = Path("checkpoints") / f"{ckpt_run}_best.pt"

            if not ckpt_path.exists():
                logger.warning("Checkpoint not found: %s — skipping.", ckpt_path)
                continue

            logger.info("Evaluating: %s (ckpt: %s)", run_name, ckpt_path)

            # Build model and load weights
            model = FusionModel(variant=variant, hidden_size=hidden_size)
            checkpoint = torch.load(
                ckpt_path, map_location=device, weights_only=True
            )
            model.load_state_dict(checkpoint["model_state_dict"])

            # Test set evaluation
            results = evaluate_on_test(
                model=model,
                test_loader=test_loader,
                device=device,
                run_name=run_name,
                log_path=LOG_PATH,
            )
            all_test_results.append(results)

            # ── Confusion matrix plot ─────────────────────────────────────────
            cm_path = plot_confusion_matrix(
                cm=results["confusion_matrix"],
                run_name=run_name,
                save_dir=PLOT_DIR,
            )
            logger.info("Saved confusion matrix: %s", cm_path)

            # ── Loss curve plot ───────────────────────────────────────────────
            history = load_history_from_log(ckpt_run)
            if history:
                lc_path = plot_loss_curves(
                    history=history,
                    run_name=run_name,
                    save_dir=PLOT_DIR,
                )
                logger.info("Saved loss curve: %s", lc_path)

    # ── Build full test results DataFrame ─────────────────────────────────────
    metric_cols = [
        "run_name", "accuracy", "precision_macro", "recall_macro",
        "f1_macro", "roc_auc",
        "precision_class0", "recall_class0", "f1_class0",
        "precision_class1", "recall_class1", "f1_class1",
        "pred_distribution",
    ]
    rows = [{k: r[k] for k in metric_cols} for r in all_test_results]
    test_df = pd.DataFrame(rows)

    # Merge with ablation val metrics
    ablation_clean = ablation_df[ablation_df["note"].fillna("") == ""].copy()
    ablation_clean["run_name"] = (
        ablation_clean["variant"] + "_h" + ablation_clean["hidden_size"].astype(str)
    )
    merged = test_df.merge(
        ablation_clean[["run_name", "variant", "hidden_size", "best_epoch",
                         "best_val_loss", "best_val_acc", "n_params"]],
        on="run_name", how="left",
    )

    # Save full results
    full_results_path = Path(RESULTS_DIR) / "test_results.csv"
    merged.drop(columns=["pred_distribution"]).to_csv(full_results_path, index=False)
    logger.info("Full test results saved to %s", full_results_path)

    # ── Comparison plots ──────────────────────────────────────────────────────
    # Add note column if missing (handles merge edge cases)
    # ── Comparison plots ──────────────────────────────────────────────────────
    if "note" not in merged.columns:
        merged["note"] = ""

    # Drop rows where variant is NaN (unmatched merge rows)
    merged = merged.dropna(subset=["variant"])
    merged["note"] = merged["note"].fillna("")

    plot_hidden_size_comparison(merged, metric="f1_macro",   save_dir=PLOT_DIR)
    plot_hidden_size_comparison(merged, metric="accuracy",   save_dir=PLOT_DIR)
    plot_ablation_heatmap(merged, save_dir=PLOT_DIR)
    plot_roc_auc_comparison(merged, save_dir=PLOT_DIR)
    logger.info("All comparison plots saved to %s/", PLOT_DIR)

    # ── Final printed summary ─────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("PHASE 5 COMPLETE — Test Set Results")
    print("=" * 75)
    display = merged[[
        "run_name", "accuracy", "f1_macro", "roc_auc",
        "f1_class0", "f1_class1", "n_params"
    ]].copy()
    display.columns = [
        "Model", "Accuracy", "F1 Macro", "ROC-AUC",
        "F1 Down", "F1 Up", "Params"
    ]
    print(display.to_string(index=False))
    print("=" * 75)

    # Best model by F1 macro
    best_row = merged.loc[merged["f1_macro"].idxmax()]
    print(f"\nBest model by F1 Macro: {best_row['run_name']}")
    print(f"  Accuracy  : {best_row['accuracy']:.4f}")
    print(f"  F1 Macro  : {best_row['f1_macro']:.4f}")
    print(f"  ROC-AUC   : {best_row['roc_auc']:.4f}")
    print(f"\nPlots saved to : {PLOT_DIR}/")
    print(f"Results saved  : {full_results_path}")


if __name__ == "__main__":
    main()