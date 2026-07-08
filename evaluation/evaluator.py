"""
evaluation/evaluator.py
------------------------
Computes full evaluation metrics on the test set for a trained model.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from utils.logger import get_logger


@torch.no_grad()
def evaluate_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    run_name: str,
    log_path: str = "logs/evaluation.log",
) -> Dict:
    """
    Run model on test set and compute all metrics.

    Returns dict with: accuracy, precision, recall, f1, roc_auc,
    confusion_matrix, classification_report, all_preds, all_labels
    """
    logger = get_logger(__name__, log_path)
    model.eval()
    model.to(device)

    all_preds = []
    all_probs = []
    all_labels = []

    for price_seq, embedding, label in test_loader:
        price_seq = price_seq.to(device)
        embedding = embedding.to(device)
        label     = label.to(device)

        logits = model(price_seq, embedding)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy = accuracy_score(all_labels, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    p_per, r_per, f1_per, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )

    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = float("nan")

    cm     = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds,
        target_names=["Down (0)", "Up (1)"],
        zero_division=0,
    )

    pred_dist = {
        "pred_0": int((all_preds == 0).sum()),
        "pred_1": int((all_preds == 1).sum()),
    }

    results = {
        "run_name":          run_name,
        "accuracy":          round(float(accuracy), 4),
        "precision_macro":   round(float(precision), 4),
        "recall_macro":      round(float(recall), 4),
        "f1_macro":          round(float(f1), 4),
        "roc_auc":           round(float(roc_auc), 4),
        "precision_class0":  round(float(p_per[0]), 4),
        "precision_class1":  round(float(p_per[1]), 4),
        "recall_class0":     round(float(r_per[0]), 4),
        "recall_class1":     round(float(r_per[1]), 4),
        "f1_class0":         round(float(f1_per[0]), 4),
        "f1_class1":         round(float(f1_per[1]), 4),
        "confusion_matrix":  cm.tolist(),
        "classification_report": report,
        "pred_distribution": pred_dist,
        "all_preds":         all_preds,
        "all_labels":        all_labels,
        "all_probs":         all_probs,
    }

    logger.info("=" * 50)
    logger.info("TEST RESULTS: %s", run_name)
    logger.info("  Accuracy  : %.4f", accuracy)
    logger.info("  Precision : %.4f (macro)", precision)
    logger.info("  Recall    : %.4f (macro)", recall)
    logger.info("  F1        : %.4f (macro)", f1)
    logger.info("  ROC-AUC   : %.4f", roc_auc)
    logger.info("  Pred dist : class0=%d, class1=%d",
                pred_dist["pred_0"], pred_dist["pred_1"])
    logger.info("\n%s", report)

    return results