"""Regression metrics shared by both pipelines."""

from __future__ import annotations

import numpy as np


def compute_regression_metrics(preds: np.ndarray, targets: np.ndarray):
    if preds.size == 0 or targets.size == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "r2": np.nan,
            "num_samples": 0,
        }

    preds = preds.reshape(-1, 1)
    targets = targets.reshape(-1, 1)

    residuals = targets - preds
    mse = float(np.mean(np.square(residuals)))
    mae = float(np.mean(np.abs(residuals)))
    rmse = np.sqrt(mse)
    nonzero_mask = np.abs(targets) > 1e-8
    if np.any(nonzero_mask):
        mape = np.mean(
            np.abs((targets[nonzero_mask] - preds[nonzero_mask]) / targets[nonzero_mask])
        ) * 100.0
    else:
        mape = np.nan

    if len(targets) > 1 and np.unique(targets).size > 1:
        ss_res = float(np.sum(np.square(residuals)))
        ss_tot = float(np.sum(np.square(targets - np.mean(targets))))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        r2 = np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "num_samples": int(targets.shape[0]),
    }
