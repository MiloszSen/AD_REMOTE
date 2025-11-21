from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor

from .utils import (
    ANOMALY_LABEL_COL,
    ANOMALY_METHOD_COL,
    ANOMALY_SCORE_COL,
    add_context_columns,
    prepare_features,
)

CANDIDATE_FEATURES: tuple[str, ...] = (
    "wartosc",
    "roll_mean_1h",
    "roll_std_1h",
    # "pf", "wsp2", "wsp3"  # na razie nie używamy wskaźników jako cech
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


def run(
    df: pd.DataFrame,
    *,
    features: Sequence[str] | None = None,
    scaler: str | None = "robust",
    **kwargs,
) -> pd.DataFrame:
    # 1. Wybór cech
    if features is None:
        feats = list(DEFAULT_FEATURES)
    else:
        feats = [f for f in features if f in df.columns]
        if not feats:
            raise ValueError(f"Żadna z podanych cech {features} nie istnieje w DataFrame.")

    base = df.copy()

    # 2. Wywalamy wiersze, gdzie w którejkolwiek cesze jest NaN
    X_raw = base.loc[:, feats].astype(float)
    mask = ~X_raw.isna().any(axis=1)

    if not mask.any():
        raise ValueError(
            f"LOF: wszystkie wiersze mają braki w cechach {feats}. "
            f"Nie ma na czym trenować."
        )

    X = prepare_features(base.loc[mask], feats, scaler=scaler)

    # 3. Trening i predykcja na podzbiorze bez NaN
    params = DEFAULT_PARAMS | kwargs
    model = LocalOutlierFactor(**params)
    labels = model.fit_predict(X)
    scores = model.negative_outlier_factor_

    # 4. Sklejamy wyniki z powrotem do pełnej ramki
    result = base.copy()
    result[ANOMALY_SCORE_COL] = np.nan
    result[ANOMALY_LABEL_COL] = 1          # zakładamy, że pominięte wiersze są "normalne"
    result.loc[mask, ANOMALY_SCORE_COL] = scores
    result.loc[mask, ANOMALY_LABEL_COL] = labels
    result[ANOMALY_METHOD_COL] = "lof"

    return add_context_columns(result)
