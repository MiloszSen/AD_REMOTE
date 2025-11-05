from pathlib import Path
import argparse
import os
import pandas as pd
import numpy as np

# ── Matplotlib w trybie bez GUI (Windows/Tk błąd) ───────────────────────────────
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from detection import DETECTORS, get_detector, get_method_alias
from detection.utils import (
    ANOMALY_LABEL_COL,
    ANOMALY_METHOD_COL,
    ANOMALY_SCORE_COL,
)

# ────────────────────────────────────────────────────────────────────────────────
# Parametry konfiguracyjne
# ────────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_MID = PROJECT_ROOT / "data" / "output_mid" / "data_long.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ustaw numer PPE do analizy; jeśli None — weź pierwszy dostępny z danych
SELECTED_PPE: str | None = 590310600000271389         # np. "590310600029270547"
# Ustaw jednostkę do analizy; None = bez filtra. Typowe: "V" (napięcie), "A" (prąd)
SELECTED_UNIT: str | None = "A"
# Ile ostatnich dni pokazać na wykresie (None = cały zakres)
PLOT_LAST_DAYS: int | None = 300
# Styl wykresu: "scatter" (zalecany przy gęstych danych) lub "line"
PLOT_STYLE: str = "line"
# Próbkowanie do wykresu (None = bez próbkowania). Np. 10 => co 10-ty punkt
DOWNSAMPLE_EVERY: int | None = 10

# Wybór metody detekcji anomalii
DETECTION_METHOD = "lof"  # dostępne: "isolation_forest", "lof"
DETECTOR_PARAMS = {
    "isolation_forest": {
        "contamination": 0.005,
        "n_estimators": 350,
        "random_state": 42,
    },
    "lof": {
        "contamination": 0.005,
        "n_neighbors": 20,
    },
}

# Post-processing (grupowanie zdarzeń)
NEAR_ZERO_THRESH = 0.1          # wartości <= tego progu traktuj jako "prawie zero" (dla V)
EVENT_GAP_MINUTES = 20          # zlepiaj anomalie oddzielone przerwą <= 20 min w jedno zdarzenie
MIN_EVENT_POINTS = 1            # odfiltruj pojedyncze strzały jako szum (ustaw 1, żeby zostawić wszystko)

# ────────────────────────────────────────────────────────────────────────────────
# Funkcje pomocnicze (ETL + sanity checks)
# ────────────────────────────────────────────────────────────────────────────────

def load_long() -> pd.DataFrame:
    if not DATA_MID.exists():
        raise FileNotFoundError(f"Brak pliku: {DATA_MID}. Uruchom preparation/seperation.py.")
    df = pd.read_parquet(DATA_MID)
    if "timestamp" not in df or "wartosc" not in df:
        raise RuntimeError("Brakuje kolumn 'timestamp' lub 'wartosc' w data_long.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def filter_by_unit(df: pd.DataFrame) -> pd.DataFrame:
    if SELECTED_UNIT is None:
        return df
    if "jednostka" not in df.columns:
        print(f"[WARN] Brak kolumny 'jednostka' — pomijam filtr jednostki.")
        return df
    d = df[df["jednostka"].astype(str) == str(SELECTED_UNIT)].copy()
    if d.empty:
        raise RuntimeError(f"Brak danych dla jednostki: {SELECTED_UNIT}")
    print(f"[INFO] Wybrano jednostke={SELECTED_UNIT}, liczba_probek={len(d)}")
    return d


def pick_one_ppe(df: pd.DataFrame) -> pd.DataFrame:
    if "numer_ppe" not in df.columns or df["numer_ppe"].isna().all():
        print("[WARN] Brak kolumny 'numer_ppe' — analizuję jedną serię łącznie.")
        return df

    ppe = SELECTED_PPE or df["numer_ppe"].dropna().astype(str).iloc[0]
    d = df[df["numer_ppe"].astype(str) == str(ppe)].copy()
    if d.empty:
        raise RuntimeError(f"Brak danych dla numer_ppe={ppe}")
    print(f"[INFO] Wybrano numer_ppe={ppe}, liczba_probek={len(d)}")
    return d


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    # sanity: usuń wiersze bez timestamp/wartości (nie trenujemy na NaN)
    d = d.dropna(subset=["timestamp", "wartosc"]).reset_index(drop=True)
    d["hour"] = d["timestamp"].dt.hour
    d["minute"] = d["timestamp"].dt.minute
    d["dayofweek"] = d["timestamp"].dt.dayofweek
    d = d.sort_values("timestamp")
    # cechy rolling w oknie ~1h (próbkowanie co 10 min => 6 próbek)
    d["roll_mean_1h"] = d["wartosc"].rolling(window=6, min_periods=1).mean()
    d["roll_std_1h"]  = d["wartosc"].rolling(window=6, min_periods=1).std().fillna(0.0)
    return d


def apply_detector(d: pd.DataFrame, method: str = DETECTION_METHOD) -> pd.DataFrame:
    detector = get_detector(method)
    overrides = DETECTOR_PARAMS.get(method, {}) or {}
    if overrides:
        print(f"[INFO] Uruchamiam detektor={method} z parametrami {overrides}")
    else:
        print(f"[INFO] Uruchamiam detektor={method}")
    return detector(d, **overrides)


def _unit_from_df(d: pd.DataFrame) -> str:
    if "jednostka" in d.columns and not d["jednostka"].dropna().empty:
        return str(d["jednostka"].dropna().astype(str).iloc[0])
    return "NA"


def _ppe_from_df(d: pd.DataFrame) -> str:
    if "numer_ppe" in d.columns and not d["numer_ppe"].dropna().empty:
        return str(d["numer_ppe"].dropna().astype(str).iloc[0])
    return "ALL"


def validate_selection(d: pd.DataFrame) -> pd.DataFrame:
    if "jednostka" in d.columns:
        units = d["jednostka"].dropna().astype(str).unique().tolist()
        if len(units) != 1:
            raise RuntimeError(f"[FAIL] Dataset zawiera wiele jednostek: {units}. Ustaw SELECTED_UNIT lub popraw filtr.")
    if "numer_ppe" in d.columns:
        ppes = d["numer_ppe"].dropna().astype(str).unique().tolist()
        if len(ppes) != 1:
            raise RuntimeError(f"[FAIL] Dataset zawiera wiele licznikow: {ppes[:5]}... Ustaw SELECTED_PPE lub popraw filtr.")
    # porządkowanie
    d = d.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
    print(f"[OK] Walidacja: 1 jednostka i 1 licznik. Rekordow={len(d)}")
    return d


def make_plot(d: pd.DataFrame, out_png: Path):
    d_plot = d.copy()

    # okno czasowe: ostatnie PLOT_LAST_DAYS
    if PLOT_LAST_DAYS is not None and len(d_plot) > 0:
        tmax = d_plot["timestamp"].max()
        tmin = tmax - pd.Timedelta(days=PLOT_LAST_DAYS)
        d_plot = d_plot[d_plot["timestamp"] >= tmin]

    # próbkowanie dla czytelności wykresu
    if DOWNSAMPLE_EVERY is not None and DOWNSAMPLE_EVERY > 1:
        d_plot = d_plot.iloc[::DOWNSAMPLE_EVERY, :]

    fig, ax = plt.subplots(figsize=(14, 5))

    # seria normalna
    if ANOMALY_LABEL_COL in d_plot.columns:
        normal_mask = d_plot[ANOMALY_LABEL_COL] == 1
    else:
        normal_mask = pd.Series(True, index=d_plot.index, dtype=bool)
    plot_style = (PLOT_STYLE or "line").lower()
    if plot_style not in {"line", "scatter"}:
        plot_style = "line"
    if plot_style == "line":
        ax.plot(
            d_plot.loc[normal_mask, "timestamp"],
            d_plot.loc[normal_mask, "wartosc"],
            lw=0.5,
            alpha=0.6,
            label="normalne",
        )
    else:
        ax.scatter(
            d_plot.loc[normal_mask, "timestamp"],
            d_plot.loc[normal_mask, "wartosc"],
            s=12,
            alpha=0.5,
            label="normalne",
        )

    # anomalie wyróżnione na czerwono
    anom = d_plot[~normal_mask]
    if not anom.empty:
        ax.scatter(anom["timestamp"], anom["wartosc"], s=15, color="red", label="anomalia")

    unit = _unit_from_df(d)
    ppe  = _ppe_from_df(d)
    if ANOMALY_METHOD_COL in d.columns and not d[ANOMALY_METHOD_COL].dropna().empty:
        method_name = d[ANOMALY_METHOD_COL].dropna().astype(str).iloc[0]
    else:
        method_name = "detector"
    title_name = method_name.replace("_", " ").title()
    ax.set_title(f"{title_name} - wartosc (unit={unit}, ppe={ppe})")
    ax.set_xlabel("czas")
    ax.set_ylabel("wartosc")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _merge_anomaly_events(d: pd.DataFrame) -> pd.DataFrame:
    if ANOMALY_LABEL_COL not in d.columns:
        return pd.DataFrame()
    x = d.copy()
    x = x.sort_values("timestamp").reset_index(drop=True)
    x["is_anom"] = (x[ANOMALY_LABEL_COL] == -1)

    x["is_near_zero"] = x["wartosc"] <= NEAR_ZERO_THRESH

    diffs = x["timestamp"].diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    base_step = diffs.median() if not diffs.empty else pd.NaT
    if pd.isna(base_step) or base_step <= pd.Timedelta(0):
        base_step = pd.Timedelta(minutes=10)

    grp = []
    cur = []
    last_ts = None
    max_gap = pd.Timedelta(minutes=EVENT_GAP_MINUTES)
    for row in x.itertuples(index=False):
        if row.is_anom:
            ts = row.timestamp
            if not cur:
                cur = [row]
            else:
                if last_ts is not None and (ts - last_ts) <= max_gap:
                    cur.append(row)
                else:
                    grp.append(cur)
                    cur = [row]
            last_ts = ts
        else:
            if cur:
                grp.append(cur)
                cur = []
                last_ts = None
    if cur:
        grp.append(cur)

    events = []
    for g in grp:
        if len(g) < MIN_EVENT_POINTS:
            continue
        start = g[0].timestamp
        end = g[-1].timestamp
        duration = (end - start) + base_step
        seg = x[(x["timestamp"] >= start) & (x["timestamp"] <= end)]
        med = seg["wartosc"].median()
        if ANOMALY_SCORE_COL in seg.columns:
            min_score = seg[ANOMALY_SCORE_COL].min()
        else:
            min_score = np.nan
        
        z_hour_mean = seg["z_hour"].mean() if "z_hour" in seg.columns else np.nan
        if (seg["is_near_zero"].mean() > 0.5) or (pd.notna(z_hour_mean) and z_hour_mean < -5):
            ev_type = "signal_loss"
        else:
            ev_type = "outlier_run"
        events.append({
            "start": start,
            "end": end,
            "duration_min": int(duration.total_seconds()//60),
            "n_points": len(seg),
            "median_value": float(med),
            "min_anomaly_score": float(min_score) if pd.notna(min_score) else np.nan,
            "type": ev_type,
        })

    return pd.DataFrame(events)


def save_outputs(d: pd.DataFrame, method: str):
    unit = _unit_from_df(d)
    ppe  = _ppe_from_df(d)
    label_series = d.get(ANOMALY_LABEL_COL)
    if label_series is not None:
        anom = d[label_series == -1].copy()
    else:
        anom = pd.DataFrame(columns=d.columns)

    method_alias = get_method_alias(method)

    out_png = OUT_DIR / f"anomalies_{method_alias}_plot_{unit}_{ppe}.png"
    make_plot(d, out_png)
    print(f"[OK] Wykres: {out_png}")

    
    events = _merge_anomaly_events(d)
    if not events.empty:
        out_evt = OUT_DIR / f"anomaly_events_{method_alias}_{unit}_{ppe}.csv"
        events.to_csv(out_evt, index=False)
        print(f"[OK] Zapisano zdarzenia: {out_evt}")
    i
    if label_series is not None:
        n_anom = int((label_series == -1).sum())
    else:
        n_anom = 0
    ratio = 100.0 * n_anom / max(1, len(d))
    print(f"[INFO] Wykryto {n_anom} anomalii ({ratio:.2f}% probek)")

    
    if ANOMALY_SCORE_COL in d.columns:
        top = d.nsmallest(10, ANOMALY_SCORE_COL)[[c for c in ["timestamp", "wartosc", ANOMALY_SCORE_COL] if c in d.columns]]
        print("Top 10 najsilniejszych anomalii:")
        with pd.option_context("display.max_rows", None, "display.width", 120):
            print(top.to_string(index=False))


def run_pipeline(method: str):
    df = load_long()
    df = filter_by_unit(df)
    d = pick_one_ppe(df)
    d = validate_selection(d)
    d = add_features(d)
    d = apply_detector(d, method)
    save_outputs(d, method)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anomaly detection pipeline")
    parser.add_argument(
        "--method",
        choices=sorted(DETECTORS),
        default=DETECTION_METHOD,
        help="Wybierz metodę detekcji anomalii",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    method = args.method or DETECTION_METHOD
    run_pipeline(method)


if __name__ == "__main__":
    main()
