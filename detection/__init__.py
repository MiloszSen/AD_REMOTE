from __future__ import annotations

from typing import Callable, Dict

import pandas as pd

from . import isolation_forest, lof, knn, savgol

DetectorFn = Callable[[pd.DataFrame], pd.DataFrame]

DETECTORS: Dict[str, DetectorFn] = {
    "isolation_forest": isolation_forest.run,
    "lof": lof.run,
    "knn": knn.run,
    "savgol": savgol.run,
}

DEFAULT_PARAMS: Dict[str, dict] = {
    "isolation_forest": isolation_forest.DEFAULT_PARAMS,
    "lof": lof.DEFAULT_PARAMS,
}

METHOD_ALIASES: Dict[str, str] = {
    "isolation_forest": "if",
    "lof": "lof",
    "knn": "knn",
    "savgol" : "sg"
}


def get_detector(name: str) -> DetectorFn:
    try:
        return DETECTORS[name]
    except KeyError as exc:
        available = ", ".join(sorted(DETECTORS))
        raise KeyError(f"Unknown detector '{name}'. Available: {available}") from exc


def get_default_params(name: str) -> dict:
    params = DEFAULT_PARAMS.get(name, {})
    return dict(params)


def get_method_alias(name: str) -> str:
    return METHOD_ALIASES.get(name, name.replace(" ", "_"))
