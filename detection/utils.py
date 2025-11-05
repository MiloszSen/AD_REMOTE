from __future__ import annotations

from typing import Iterable

import pandas as pd
import numpy as np

ANOMALY_SCORE_COL = "anomaly_score"
ANOMALY_LABEL_COL = "anomaly_label"
ANOMALY_METHOD_COL = "anomaly_method"


def prepare_features(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise KeyError(f"Missing features for detector: {missing}")
    return df[list(features)].astype(float)


def add_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "wartosc" in result.columns:
        result["prev_value"] = result["wartosc"].shift(1)
        result["delta"] = result["wartosc"] - result["prev_value"]
        if {"roll_mean_1h", "roll_std_1h"}.issubset(result.columns):
            eps = 1e-9
            result["z_roll"] = (result["wartosc"] - result["roll_mean_1h"]) / (
                result["roll_std_1h"].replace(0, np.nan) + eps
            )
    if "hour" in result.columns and "wartosc" in result.columns:
        hour_stats = (
            result.groupby("hour")["wartosc"]
            .agg(hour_mean="mean", hour_std="std")
            .join(result.groupby("hour")["wartosc"].quantile(0.05).rename("hour_p05"))
            .join(result.groupby("hour")["wartosc"].quantile(0.95).rename("hour_p95"))
        ).reset_index()
        result = result.merge(hour_stats, on="hour", how="left")
        result["z_hour"] = (result["wartosc"] - result["hour_mean"]) / result["hour_std"].replace(0, np.nan)
    return result
