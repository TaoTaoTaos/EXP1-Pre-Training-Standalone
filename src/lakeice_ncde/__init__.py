"""Standalone Neural CDE package for lake-ice transfer experiments."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

from lakeice_ncde.config import load_config

__all__ = ["load_config"]
