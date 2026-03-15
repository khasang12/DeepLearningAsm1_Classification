"""Logging helpers — Console + TensorBoard."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

_loggers: dict[str, logging.Logger] = {}
_tb_writer: SummaryWriter | None = None


def setup_logger(
    name: str = "dl-assignment1",
    log_dir: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create (or retrieve) a named logger with console + optional file handler.

    Parameters
    ----------
    name : str
        Logger name.
    log_dir : str | Path | None
        If provided, a ``FileHandler`` is added writing to ``<log_dir>/<name>.log``.
    level : int
        Logging level.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}.log")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "dl-assignment1") -> logging.Logger:
    """Retrieve an existing logger (creates one with defaults if needed)."""
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


def get_tb_writer(log_dir: str | Path = "outputs/logs") -> "SummaryWriter":
    """Lazily create a TensorBoard SummaryWriter."""
    global _tb_writer
    if _tb_writer is None:
        from torch.utils.tensorboard import SummaryWriter

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        _tb_writer = SummaryWriter(log_dir=str(log_dir))
    return _tb_writer


def close_tb_writer() -> None:
    """Flush and close the TensorBoard writer."""
    global _tb_writer
    if _tb_writer is not None:
        _tb_writer.flush()
        _tb_writer.close()
        _tb_writer = None
