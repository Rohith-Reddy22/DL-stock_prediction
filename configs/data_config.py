"""
configs/data_config.py
----------------------
Central configuration for the data pipeline (Phase 1).
All tuneable parameters live here — no magic numbers scattered across modules.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    # ── Ticker & date range ───────────────────────────────────────────────────
    ticker: str = "AAPL"
    start_date: str = "2018-01-01"
    end_date: str = "2020-06-04"

    # ── Technical indicator windows ───────────────────────────────────────────
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    sma_windows: List[int] = field(default_factory=lambda: [10, 20])
    ema_windows: List[int] = field(default_factory=lambda: [10, 20])

    # ── Feature columns (must match indicator output exactly) ─────────────────
    feature_cols: List[str] = field(default_factory=lambda: [
        "Open", "High", "Low", "Close", "Volume",
        "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "SMA10", "SMA20", "EMA10", "EMA20",
    ])

    # ── Label generation ──────────────────────────────────────────────────────
    # ε threshold: label=1 only if Close(T) > Close(T-1) * (1 + label_threshold)
    # Set to 0.0 for strict spec compliance; 0.002 recommended to reduce noise.
    label_threshold: float = 0.0

    # ── Chronological split ratios ────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # test_ratio is implicitly 1 - train_ratio - val_ratio = 0.15

    # ── LSTM sequence window ──────────────────────────────────────────────────
    sequence_length: int = 30      # days of history used as input (T-30 to T-1)

    # ── Paths ─────────────────────────────────────────────────────────────────
    raw_data_path: str = "data/raw/raw_ohlcv.csv"
    processed_data_path: str = "data/processed/processed_features.csv"
    aligned_data_path: str = "data/processed/aligned_dataset.csv"
    scaler_path: str = "checkpoints/scaler.pkl"
    class_weights_path: str = "configs/class_weights.pt"
    split_info_path: str = "configs/split_info.json"
    log_path: str = "logs/phase1.log"

    # ── Class imbalance warning threshold ────────────────────────────────────
    # Warn if majority class exceeds this fraction of the training set.
    imbalance_warn_threshold: float = 0.60