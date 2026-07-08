"""
training/trainer.py
--------------------
Training loop for a single model run.

Features:
- WeightedCrossEntropyLoss from Phase 1 class weights
- Adam optimiser with ReduceLROnPlateau scheduler
- EarlyStopping on validation loss
- Best model checkpoint saved per run
- Per-epoch metrics logged to file
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from utils.logger import get_logger


@dataclass
class TrainerConfig:
    """All hyperparameters for a single training run."""
    # Optimiser
    learning_rate: float = 0.001
    weight_decay: float = 0.0

    # Scheduler
    lr_factor: float = 0.5
    lr_patience: int = 5
    lr_min: float = 1e-6

    # Early stopping
    early_stop_patience: int = 15

    # Training
    max_epochs: int = 100
    device: str = "cpu"

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_path: str = "logs/training.log"

    # Run identity (set per experiment)
    run_name: str = "model"


@dataclass
class EpochMetrics:
    """Metrics recorded for a single epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    lr: float


class EarlyStopping:
    """
    Stops training when validation loss hasn't improved for `patience` epochs.

    Args:
        patience:  Epochs to wait before stopping.
        min_delta: Minimum improvement to count as progress.
    """

    def __init__(self, patience: int = 15, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float) -> bool:
        """
        Call after each epoch.

        Returns:
            True if training should stop, False otherwise.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0

    for price_seq, embedding, label in loader:
        price_seq = price_seq.to(device)
        embedding = embedding.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logits = model(price_seq, embedding)
        loss = criterion(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ← add this
        optimizer.step()

        total_loss += loss.item() * len(label)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model on a DataLoader. Returns (mean_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0

    for price_seq, embedding, label in loader:
        price_seq = price_seq.to(device)
        embedding = embedding.to(device)
        label = label.to(device)

        logits = model(price_seq, embedding)
        loss = criterion(logits, label)

        total_loss += loss.item() * len(label)
        preds = logits.argmax(dim=1)
        correct += (preds == label).sum().item()

    n = len(loader.dataset)
    return total_loss / n, correct / n


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: torch.Tensor,
    config: TrainerConfig,
) -> Tuple[nn.Module, List[EpochMetrics], str]:
    """
    Full training loop for one experiment run.

    Args:
        model:         FusionModel instance (any variant/hidden_size).
        train_loader:  Training DataLoader.
        val_loader:    Validation DataLoader.
        class_weights: 1-D FloatTensor [w0, w1] from Phase 1.
        config:        TrainerConfig with all hyperparameters.

    Returns:
        Tuple of:
        - model loaded with best checkpoint weights
        - list of EpochMetrics for each epoch
        - path to saved best checkpoint
    """
    logger = get_logger(__name__, config.log_path)
    device = torch.device(config.device)

    model = model.to(device)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
        min_lr=config.lr_min,
    )
    early_stopping = EarlyStopping(patience=config.early_stop_patience)

    # Checkpoint path for this run
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = str(ckpt_dir / f"{config.run_name}_best.pt")

    best_val_loss = float("inf")
    history: List[EpochMetrics] = []

    logger.info("Starting training: %s | device=%s | params=%d",
                config.run_name, device, sum(p.numel() for p in model.parameters() if p.requires_grad))

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=current_lr,
        )
        history.append(metrics)

        logger.info(
            "[%s] Epoch %3d | train_loss=%.4f | val_loss=%.4f | "
            "val_acc=%.4f | lr=%.2e",
            config.run_name, epoch, train_loss, val_loss, val_acc, current_lr,
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "run_name": config.run_name,
            }, best_ckpt_path)
            logger.info("  ✓ New best checkpoint saved (val_loss=%.4f)", val_loss)

        # Early stopping check
        if early_stopping.step(val_loss):
            logger.info(
                "[%s] Early stopping at epoch %d (no improvement for %d epochs).",
                config.run_name, epoch, config.early_stop_patience,
            )
            break

    # Load best weights before returning
    checkpoint = torch.load(best_ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(
        "[%s] Training complete. Best val_loss=%.4f at epoch %d.",
        config.run_name, best_val_loss, checkpoint["epoch"],
    )

    return model, history, best_ckpt_path