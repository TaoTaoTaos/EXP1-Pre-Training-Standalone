from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.logging import RichHandler


def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Create a logger with Rich console output and optional file logging."""
    logger = logging.getLogger("lakeice_ncde")
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    console_handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(file_handler)

    logging.captureWarnings(True)
    sys.excepthook = _exception_hook(logger)
    return logger


def _exception_hook(logger: logging.Logger):
    def handler(exc_type, exc_value, exc_traceback) -> None:
        logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    return handler
