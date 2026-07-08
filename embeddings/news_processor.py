"""
embeddings/news_processor.py
-----------------------------
Loads the raw financial news CSV, normalises dates, and groups all headlines
per trading day into a single concatenated text block.

Design decisions:
- Uses all stocks (not just AAPL) as market-wide sentiment signal. This is
  legitimate for predicting a single stock — macro market sentiment affects
  all equities.
- Headlines are joined with " [SEP] " so FinBERT's tokenizer sees them as
  logically separate sentences rather than one run-on string.
- Mixed timezone handling: the CSV contains both timezone-aware and naive
  datetime strings. format='mixed' + utc=True handles both safely.
- We do NOT filter by stock ticker — every headline on a given date
  contributes to that day's market sentiment embedding.
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from utils.logger import get_logger


def load_and_group_news(
    news_csv_path: str,
    start_date: str,
    end_date: str,
    log_path: str = "logs/phase2.log",
) -> Dict[str, str]:
    """
    Load the news CSV and group all headlines by normalised date.

    Args:
        news_csv_path: Path to the raw financial news CSV file.
                       Expected columns: ['headline', 'date', 'stock']
        start_date:    Filter start date string 'YYYY-MM-DD' (inclusive).
        end_date:      Filter end date string 'YYYY-MM-DD' (inclusive).
        log_path:      Path for log output.

    Returns:
        Dict mapping date string 'YYYY-MM-DD' → concatenated headline text.
        Only dates within [start_date, end_date] are included.

    Raises:
        FileNotFoundError: If the news CSV does not exist.
        ValueError:        If required columns are missing.
    """
    logger = get_logger(__name__, log_path)

    path = Path(news_csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"News CSV not found at '{news_csv_path}'. "
            "Run the data download step first."
        )

    logger.info("Loading news CSV from %s ...", news_csv_path)
    df = pd.read_csv(
        news_csv_path,
        usecols=["headline", "date", "stock"],
        dtype={"headline": str, "stock": str},
    )
    logger.info("Loaded %d raw rows.", len(df))

    # ── Date normalisation ────────────────────────────────────────────────────
    # Mixed formats: "2020-06-05 10:30:00-04:00" and "2020-05-22 00:00:00"
    df["date"] = (
        pd.to_datetime(df["date"], format="mixed", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )

    # ── Filter to our window ──────────────────────────────────────────────────
    mask = (df["date"] >= pd.Timestamp(start_date)) & (
        df["date"] <= pd.Timestamp(end_date)
    )
    df = df[mask].copy()
    logger.info(
        "After date filter (%s → %s): %d rows remaining.",
        start_date, end_date, len(df),
    )

    # ── Drop nulls and empty headlines ───────────────────────────────────────
    df["headline"] = df["headline"].fillna("").str.strip()
    df = df[df["headline"] != ""]
    logger.info("After dropping empty headlines: %d rows.", len(df))

    # ── Group by date → concatenate headlines ────────────────────────────────
    # Join with [SEP] so FinBERT sees sentence boundaries
    grouped = (
        df.groupby("date")["headline"]
        .apply(lambda headlines: " [SEP] ".join(headlines.tolist()))
        .to_dict()
    )

    # Convert Timestamp keys to 'YYYY-MM-DD' strings
    grouped = {k.strftime("%Y-%m-%d"): v for k, v in grouped.items()}

    logger.info(
        "Grouped into %d unique dates. "
        "Avg headlines/day: %.1f",
        len(grouped),
        df.groupby("date").size().mean(),
    )

    # Log a few sample dates
    sample_dates = sorted(grouped.keys())[:3]
    for d in sample_dates:
        text = grouped[d]
        preview = text[:120] + "..." if len(text) > 120 else text
        logger.debug("Sample [%s]: %s", d, preview)

    return grouped