# detection/comparison.py

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def _spearman_corr(s1: pd.Series, s2: pd.Series) -> float:
    """
    Spearman liczymy jako korelację Pearsona na rangach.
    Pomijamy pary, gdzie któraś wartość jest NaN.
    """
    s1 = pd.Series(s1)
    s2 = pd.Series(s2)
    mask = s1.notna() & s2.notna()
    if mask.sum() < 2:
        return np.nan

    r1 = s1[mask].rank(method="average")
    r2 = s2[mask].rank(method="average")
    return r1.corr(r2)  


def build_comparison_report(
    df: pd.DataFrame,
    methods: Sequence[str] = ("lof", "knn", "isolation_forest"),
    score_suffix: str = "_score",
    label_suffix: str = "_label",
) -> pd.DataFrame:
    """
    Buduje zwartą tabelę porównawczą dla zadanych metod.
    Oczekuje w df kolumn:
        '{method}_score'  – ciągły anomaly_score
        '{method}_label'  – 1 (normalny), -1 (anomalia)

    Zwraca DataFrame z wierszem dla każdej pary metod:
        method_1, method_2
        spearman_score       – korelacja Spearmana anomaly_score
        agreement_pct        – procent zgodności etykiet 1 / -1
        jaccard_anom         – Jaccard dla zbiorów anomalii
        n_anom_1, n_anom_2   – liczba anomalii dla każdej metody
        n_anom_both          – liczba anomalii wspólnych
    """
    rows: list[dict] = []

    for m1, m2 in itertools.combinations(methods, 2):
        s1 = df.get(f"{m1}{score_suffix}")
        s2 = df.get(f"{m2}{score_suffix}")
        l1 = df.get(f"{m1}{label_suffix}")
        l2 = df.get(f"{m2}{label_suffix}")

        if s1 is None or s2 is None or l1 is None or l2 is None:
           
            continue

       
        rho = _spearman_corr(s1, s2)

        mask_valid = l1.notna() & l2.notna()
        same = (l1[mask_valid] == l2[mask_valid])
        agreement_pct = float(same.mean() * 100.0) if mask_valid.any() else np.nan

        mask1 = (l1 == -1)
        mask2 = (l2 == -1)
        inter = int((mask1 & mask2).sum())
        union = int((mask1 | mask2).sum())
        jaccard = float(inter / union) if union > 0 else np.nan

        n_anom_1 = int(mask1.sum())
        n_anom_2 = int(mask2.sum())

        rows.append(
            {
                "method_1": m1,
                "method_2": m2,
                "spearman_score": rho,
                "agreement_pct": agreement_pct,
                "jaccard_anom": jaccard,
                "n_anom_1": n_anom_1,
                "n_anom_2": n_anom_2,
                "n_anom_both": inter,
            }
        )

    report = pd.DataFrame(rows)
    
    cols = [
        "method_1",
        "method_2",
        "spearman_score",
        "agreement_pct",
        "jaccard_anom",
        "n_anom_1",
        "n_anom_2",
        "n_anom_both",
    ]
    return report.reindex(columns=cols)
