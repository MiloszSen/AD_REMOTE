
from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from .utils import (
    ANOMALY_LABEL_COL, ANOMALY_METHOD_COL, ANOMALY_SCORE_COL,
    add_context_columns, prepare_features
)

DEFAULT_FEATURES: Sequence[str] = ("wartosc", "roll_mean_1h", "roll_std_1h")
DEFAULT_PARAMS = {"n_neighbors": 20, "metric": "minkowski", "p": 2, "n_jobs": -1}

def run(df: pd.DataFrame, *, features: Sequence[str] | None = None,
        scaler: str | None = "robust", contamination: float = 0.005, **kwargs) -> pd.DataFrame:
    feats = features or DEFAULT_FEATURES
    X = prepare_features(df, feats, scaler=scaler)

    params = DEFAULT_PARAMS | {"n_neighbors": int(kwargs.get("n_neighbors", DEFAULT_PARAMS["n_neighbors"]))}
    nn = NearestNeighbors(n_neighbors=params["n_neighbors"],
                          metric=params["metric"], p=params["p"], n_jobs=params["n_jobs"])
    nn.fit(X)
    dists, _ = nn.kneighbors(X)
    score = dists.mean(axis=1)            
    thr = np.quantile(score, 1.0 - float(contamination))
    labels = np.where(score >= thr, -1, 1)

    res = df.copy()
    
    res[ANOMALY_SCORE_COL]  = -score
    res[ANOMALY_LABEL_COL]  = labels
    res[ANOMALY_METHOD_COL] = "knn"
    return add_context_columns(res)
