from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

ANOMALY_SCORE_COL = "anomaly_score"
ANOMALY_LABEL_COL = "anomaly_label"
ANOMALY_METHOD_COL = "anomaly_method"


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


def get_scaler(name: str | None):
    if name in (None, "none"):
        return None
    table = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    if name not in table:
        raise ValueError(f"Unknown scaler: {name}")
    return table[name]


def choose_features(df: pd.DataFrame, candidates: Sequence[str]) -> list[str]:
    """
    Bierzemy tylko te cechy, które:
      - istnieją w df.columns
      - NIE są w całości NaN
    """
    feats: list[str] = []
    for c in candidates:
        if c in df.columns:
            col = df[c]
            if not col.isna().all():
                feats.append(c)

    if not feats:
        raise ValueError(f"None of the candidate features exist: {candidates}")

    return feats


def prepare_features(df: pd.DataFrame, feats: Sequence[str], scaler: str | None = None) -> np.ndarray:
    X = df.loc[:, feats].astype(float).values
    sc = get_scaler(scaler)
    return sc.fit_transform(X) if sc else X
