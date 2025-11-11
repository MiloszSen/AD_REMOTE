from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.ensemble import IsolationForest

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
    "n_estimators": 350,
    "contamination": 0.005,
    "random_state": 42,
    "bootstrap": False,
    "n_jobs": -1,
}


def run(df: pd.DataFrame, *, features=None, scaler: str | None = "robust", **kwargs) -> pd.DataFrame:
    feats = features or DEFAULT_FEATURES
    X = prepare_features(df, feats, scaler=scaler)
    params = DEFAULT_PARAMS | kwargs
    model = IsolationForest(**params)
    model.fit(X)

    result = df.copy()
    result[ANOMALY_SCORE_COL] = model.decision_function(X)
    result[ANOMALY_LABEL_COL] = model.predict(X)
    result[ANOMALY_METHOD_COL] = "isolation_forest"

    return add_context_columns(result)
