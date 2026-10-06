"""Shared settings for the fair benchmark."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = Path(os.environ.get("BENCH_RESULTS", PROJECT_ROOT / "results" / "full_sg"))
# Precomputed tensors are large (several GB per task); keep them off the
# project disk unless told otherwise.
CACHE_ROOT = Path(os.environ.get("BENCH_CACHE", RESULT_ROOT / "cache"))

EDGE_SUBSET = ("same_mrt", "similar_price")
SIMILAR_PRICE_K = 5
HORIZONS = (1, 12, 24)
WINDOW = 12
VALIDATION_MONTHS = 24
