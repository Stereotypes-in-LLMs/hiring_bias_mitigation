"""Uniform logging for every entry point."""

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("HBM_LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    return logger
