from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from .utils import (
    ANOMALY_LABEL_COL,
    ANOMALY_METHOD_COL,
    ANOMALY_SCORE_COL,
    add_context_columns,
    prepare_features,
)

DEFAULT_FEATURES: Sequence[str] = (
    "wartosc",
    "roll_mean_1h",
    "roll_std_1h",
)

DEFAULT_PARAMS = {
    "n_neighbors": 20,
    "contamination": 0.005,
    "algorithm": "auto",
    "leaf_size": 30,
    "metric": "minkowski",
    "p": 2,
    "metric_params": None,
    "n_jobs": -1,
}


def run(df: pd.DataFrame, *, features: Sequence[str] | None = None, **kwargs) -> pd.DataFrame:
    feats = features or DEFAULT_FEATURES
    X = prepare_features(df, feats)

    params = DEFAULT_PARAMS | kwargs
    model = LocalOutlierFactor(**params)
    labels = model.fit_predict(X)
    scores = model.negative_outlier_factor_

    result = df.copy()
    result[ANOMALY_SCORE_COL] = scores
    result[ANOMALY_LABEL_COL] = labels
    result[ANOMALY_METHOD_COL] = "lof"

    return add_context_columns(result)
