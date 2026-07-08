"""
utils/logger.py
---------------
Centralised logging setup.
Call get_logger(__name__) in every module — logs go to both console and file.
"""

import logging
import os
from pathlib import Path


def get_logger(name: str, log_path: str = "logs/phase1.log") -> logging.Logger:
    """
    Return a logger that writes to both stdout and a rotating log file.

    Args:
        name:     Module name, typically __name__.
        log_path: Path to the log file. Parent directory is created if needed.

    Returns:
        Configured Logger instance.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured — avoid duplicate handlers

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO and above)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler (DEBUG and above)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger