"""
embeddings/embedding_pipeline.py
----------------------------------
Orchestrates Phase 2: generates one FinBERT embedding per trading day
and caches them to disk.

Output structure:
    embeddings/
        2018-01-31.pt       ← torch.FloatTensor of shape (768,)
        2018-02-01.pt
        ...
    embeddings/embedding_index.json   ← {"2018-01-31": "embeddings/2018-01-31.pt", ...}

Design decisions:
- One .pt file per date: fast random access in Phase 3 Dataset.__getitem__
- embedding_index.json maps date strings to file paths for O(1) lookup
- Progress tracked with tqdm
- Coverage report at the end shows how many trading days have embeddings
- Skips already-computed embeddings (idempotent) — safe to re-run
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from embeddings.finbert_encoder import FinBERTEncoder
from embeddings.news_processor import load_and_group_news
from utils.logger import get_logger


def generate_embeddings(
    news_csv_path: str,
    trading_dates: List[str],
    start_date: str,
    end_date: str,
    output_dir: str = "embeddings",
    device: Optional[torch.device] = None,
    log_path: str = "logs/phase2.log",
) -> Dict[str, str]:
    """
    Generate and cache FinBERT embeddings for all trading dates.

    Args:
        news_csv_path:  Path to raw financial news CSV.
        trading_dates:  List of 'YYYY-MM-DD' strings from aligned_dataset.
                        These are the dates we NEED embeddings for.
        start_date:     News filter start date.
        end_date:       News filter end date.
        output_dir:     Directory to save .pt embedding files.
        device:         torch.device for FinBERT inference.
        log_path:       Path for log output.

    Returns:
        Dict mapping date string → path to .pt file.
        Dates with no news are skipped and NOT included in the index.
    """
    logger = get_logger(__name__, log_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    index_path = out_path / "embedding_index.json"

    # ── Load existing index (idempotent re-runs) ──────────────────────────────
    if index_path.exists():
        with open(index_path) as f:
            embedding_index = json.load(f)
        logger.info(
            "Loaded existing embedding index with %d entries.", len(embedding_index)
        )
    else:
        embedding_index = {}

    # ── Load and group news headlines ─────────────────────────────────────────
    date_to_text = load_and_group_news(
        news_csv_path=news_csv_path,
        start_date=start_date,
        end_date=end_date,
        log_path=log_path,
    )

    # Only process dates we actually need (trading days) that have news
    dates_with_news = [d for d in trading_dates if d in date_to_text]
    dates_already_done = [d for d in dates_with_news if d in embedding_index]
    dates_to_process = [d for d in dates_with_news if d not in embedding_index]

    logger.info(
        "Trading dates needed : %d", len(trading_dates)
    )
    logger.info(
        "Dates with news      : %d (%.1f%%)",
        len(dates_with_news),
        len(dates_with_news) / len(trading_dates) * 100,
    )
    logger.info(
        "Already cached       : %d", len(dates_already_done)
    )
    logger.info(
        "To generate          : %d", len(dates_to_process)
    )

    if not dates_to_process:
        logger.info("All embeddings already cached. Nothing to do.")
        return embedding_index

    # ── Load FinBERT ──────────────────────────────────────────────────────────
    encoder = FinBERTEncoder(device=device, log_path=log_path)

    # ── Generate embeddings ───────────────────────────────────────────────────
    logger.info("Generating %d embeddings...", len(dates_to_process))

    for date_str in tqdm(sorted(dates_to_process), desc="Encoding", unit="day"):
        text = date_to_text[date_str]

        try:
            embedding = encoder.encode(text)  # shape: (768,)

            # Validate shape
            assert embedding.shape == (768,), (
                f"Unexpected embedding shape {embedding.shape} for date {date_str}"
            )

            # Save to disk
            save_path = out_path / f"{date_str}.pt"
            torch.save(embedding, save_path)

            embedding_index[date_str] = str(save_path)

        except Exception as e:
            logger.error("Failed to encode date %s: %s", date_str, e)
            continue

    # ── Save updated index ────────────────────────────────────────────────────
    with open(index_path, "w") as f:
        json.dump(embedding_index, f, indent=2, sort_keys=True)
    logger.info("Embedding index saved to %s", index_path)

    # ── Coverage report ───────────────────────────────────────────────────────
    covered = [d for d in trading_dates if d in embedding_index]
    missing = [d for d in trading_dates if d not in embedding_index]

    logger.info("=" * 50)
    logger.info("EMBEDDING COVERAGE REPORT")
    logger.info("  Trading days total : %d", len(trading_dates))
    logger.info("  Days with embedding: %d (%.1f%%)",
                len(covered), len(covered) / len(trading_dates) * 100)
    logger.info("  Days missing       : %d", len(missing))

    if missing:
        logger.warning(
            "Missing embeddings for %d dates — these rows will be "
            "excluded from the dataset in Phase 3.",
            len(missing),
        )
        logger.debug("Missing dates: %s", missing[:10])

    return embedding_index