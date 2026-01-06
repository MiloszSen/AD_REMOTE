from pathlib import Path
import argparse
import os
import pandas as pd
import numpy as np
from detection.indicators import compute_S, compute_pf, compute_wsp2, compute_wsp3, flags_from_indicators
os.environ.setdefault("MPLBACKEND", "Agg")
from preparation.load_data import ppe
from detection.comparison import build_comparison_report
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



SELECTED_PPE: str | None = ppe[25254]

SELECTED_UNIT: str | None = None

PLOT_LAST_DAYS: int | None = None

PLOT_STYLE: str = "line"

DOWNSAMPLE_EVERY: int | None = None

contamin = 0.01
DETECTION_METHOD = "knn"  

DETECTOR_PARAMS = {
    "isolation_forest": {
        "contamination": contamin,
        "n_estimators": 350,
        "random_state": 42,
        "scaler": "robust",
    },
    "lof": {
        "contamination": contamin,
        "n_neighbors": 80,
        "scaler": "robust",
    },
    "knn": {
        "contamination": contamin,
        "n_neighbors": 80,
        "scaler": "robust",
    },
    "savgol": {
    "window_length": 11,   
    "polyorder": 2,
    "z_thr": 4.0,
    "contamination": contamin,
}
}

COMPARISON_METHODS = [
    m for m in ("lof", "knn", "isolation_forest")
    if m in DETECTORS
]


def _spearman_corr(s1: pd.Series, s2: pd.Series) -> float:
    s1 = pd.Series(s1)
    s2 = pd.Series(s2)
    mask = s1.notna() & s2.notna()
    if mask.sum() < 2:
        return np.nan

    r1 = s1[mask].rank(method="average")
    r2 = s2[mask].rank(method="average")
    return r1.corr(r2)  

NEAR_ZERO_THRESH = 0.1          
EVENT_GAP_MINUTES = 30          
MIN_EVENT_POINTS = 20       

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

    d = d.dropna(subset=["timestamp"]).reset_index(drop=True)

    d["hour"] = d["timestamp"].dt.hour
    d["minute"] = d["timestamp"].dt.minute
    d["dayofweek"] = d["timestamp"].dt.dayofweek
    d = d.sort_values("timestamp")

    
    d["roll_mean_1h"] = d["wartosc"].rolling(window=6, min_periods=1).mean()
    d["roll_std_1h"]  = d["wartosc"].rolling(window=6, min_periods=1).std().fillna(0.0)

    hhmm = d["timestamp"].dt.strftime("%H:%M")
    grp = d.groupby(hhmm)["wartosc"]
    med = grp.transform("median")
    iqr = grp.transform(lambda x: x.quantile(0.75) - x.quantile(0.25)).replace(0, np.nan)
    d["z_hour"] = (d["wartosc"] - med) / iqr
    d["z_hour"] = d["z_hour"].fillna(0.0)

    return d

def enrich_physics(d: pd.DataFrame) -> pd.DataFrame:
    x = d.copy()
    
    hr = x["timestamp"].dt.hour
    x["is_night"] = (hr >= 22) | (hr < 4)

    x["S"]    = compute_S(x)
    x["pf"]   = compute_pf(x)
    x["wsp2"] = compute_wsp2(x)
    x["wsp3"] = compute_wsp3(x)

    flags = flags_from_indicators(x)
    x = pd.concat([x, flags.add_prefix("flag_")], axis=1)
    return x

def apply_detector(d: pd.DataFrame, method: str = DETECTION_METHOD) -> pd.DataFrame:
    detector = get_detector(method)

    
    overrides = DETECTOR_PARAMS.get(method, {}) or {}
    overrides = overrides.copy()  

    
    if method in {"lof", "isolation_forest", "knn"} and SELECTED_UNIT is None:
        
        indicator_feats = [
            c for c in ["S", "wsp2", "wsp3"]
            if c in d.columns and not d[c].isna().all()
        ]
        if indicator_feats:
            overrides["features"] = indicator_feats


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
   
    if "numer_ppe" in d.columns:
        ppes = d["numer_ppe"].dropna().astype(str).unique().tolist()
        if len(ppes) != 1:
            raise RuntimeError(f"[FAIL] Dataset zawiera wiele licznikow: {ppes[:5]}... Ustaw SELECTED_PPE lub popraw filtr.")

   
    if "jednostka" in d.columns and SELECTED_UNIT is not None:
        units = d["jednostka"].dropna().astype(str).unique().tolist()
        if len(units) != 1:
            raise RuntimeError(f"[FAIL] Dataset zawiera wiele jednostek: {units}. Ustaw SELECTED_UNIT lub popraw filtr.")

    d = d.sort_values("timestamp").reset_index(drop=True)
    print(f"[OK] Walidacja: 1 licznik. Jednostki={sorted(d['jednostka'].dropna().astype(str).unique().tolist())}")
    return d

import unicodedata
import re

def reshape_three_phase(d: pd.DataFrame) -> pd.DataFrame:
    
    d = d.copy()

    needed_cols = [
        "timestamp",
        "wartosc",
        "wielkosc_mierzona",
        "typ_wielkosci_mierzonej",
        "jednostka",
    ]
    missing = [c for c in needed_cols if c not in d.columns]
    if missing:
        raise RuntimeError(f"Brakuje kolumn do reshape_three_phase: {missing}")

    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")

    
    def _norm_series(s: pd.Series) -> pd.Series:
        s2 = s.astype(str).str.replace("ł", "l").str.replace("Ł", "L")
        s2 = s2.apply(
            lambda name: "".join(
                c
                for c in unicodedata.normalize("NFKD", name)
                if not unicodedata.combining(c)
            )
        )
        s2 = s2.str.lower().str.strip()
        s2 = s2.str.replace(r"[^\w:]+", "_", regex=True)
        s2 = s2.str.replace(r"_+", "_", regex=True).str.strip("_")
        return s2

    d["wielkosc_norm"] = _norm_series(d["wielkosc_mierzona"])
    d["typ_norm"]      = _norm_series(d["typ_wielkosci_mierzonej"])
    d["unit_norm"]     = _norm_series(d["jednostka"])

    d["feat_name"] = (
        d["wielkosc_norm"] + "__" + d["typ_norm"] + "__" + d["unit_norm"]
    )

    wide = (
        d.pivot_table(
            index="timestamp",
            columns="feat_name",
            values="wartosc",
            aggfunc="mean",
        )
        .sort_index()
        .reset_index()
    )
    rename_map = {
        
        "prad_elektryczny__fazy_1__a": "I1",
        "prad_elektryczny__fazy_2__a": "I2",
        "prad_elektryczny__fazy_3__a": "I3",

        "napiecie_elektryczne__fazy_1__v": "U1",
        "napiecie_elektryczne__fazy_2__v": "U2",
        "napiecie_elektryczne__fazy_3__v": "U3",
    }

    wide = wide.rename(columns=rename_map)

    
    for col in ["numer_ppe", "typ_urzadzenia", "system_odczytowy"]:
        if col in d.columns:
            val = d[col].dropna()
            wide[col] = str(val.iloc[0]) if not val.empty else np.nan

    if {"I1", "I2", "I3"}.issubset(wide.columns):
        wide["wartosc"] = wide[["I1", "I2", "I3"]].abs().sum(axis=1)
    else:
        wide["wartosc"] = np.nan

    return wide


def make_plot(d: pd.DataFrame, out_png: Path):
    d_plot = d.copy()

    if PLOT_LAST_DAYS is not None and len(d_plot) > 0:
        tmax = d_plot["timestamp"].max()
        tmin = tmax - pd.Timedelta(days=PLOT_LAST_DAYS)
        d_plot = d_plot[d_plot["timestamp"] >= tmin]

    if DOWNSAMPLE_EVERY is not None and DOWNSAMPLE_EVERY > 1:
        d_plot = d_plot.iloc[::DOWNSAMPLE_EVERY, :]

    fig, ax = plt.subplots(figsize=(14, 5))

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

def classify_fault_single(seg: pd.DataFrame) -> str:
    
    z_hour_mean = seg["z_hour"].mean() if "z_hour" in seg.columns else np.nan
    is_near_zero_mean = seg["is_near_zero"].mean() if "is_near_zero" in seg.columns else 0.0

    if (is_near_zero_mean > 0.5) or (pd.notna(z_hour_mean) and z_hour_mean < -5):
        return "signal_loss"
    else:
        return "outlier_run"


def classify_fault_multi(seg: pd.DataFrame) -> str:

    def med(col):
        return seg[col].median() if col in seg.columns else np.nan

    def mx(col):
        return seg[col].max() if col in seg.columns else np.nan

    S_med  = med("S")
    wsp2_max = mx("wsp2")
    wsp3_max = mx("wsp3")

    U_med = {f"U{i}": med(f"U{i}") for i in (1, 2, 3)}
    I_med = {f"I{i}": med(f"I{i}") for i in (1, 2, 3)}

    for ph in (1, 2, 3):
        u = U_med.get(f"U{ph}", np.nan)
        if pd.notna(u) and u < 50:
            return f"phase_loss_L{ph}"

    for ph in (1, 2, 3):
        u = U_med.get(f"U{ph}", np.nan)
        i = I_med.get(f"I{ph}", np.nan)
        if pd.notna(u) and pd.notna(i):
            if u > 150 and i < 0.1:
                return f"current_phase_loss_L{ph}"

    if pd.notna(wsp2_max):
        if wsp2_max > 10:
            return "voltage_unbalance_critical"
        if wsp2_max > 5:
            return "voltage_unbalance"

    if pd.notna(wsp3_max):
        if wsp3_max > 15:
            return "zero_sequence_critical"
        if wsp3_max > 10:
            return "zero_sequence"

    if pd.notna(S_med):
        if S_med > 3000:
            return "overload_critical"
        if S_med > 1500:
            return "overload"

    for ph in (1, 2, 3):
        i = I_med.get(f"I{ph}", np.nan)
        if pd.notna(i) and i < -0.2:
            return "reverse_current"

    if pd.notna(S_med) and S_med < 5:
        if all((pd.notna(U_med.get(f"U{i}", np.nan)) and U_med[f"U{i}"] > 150) for i in (1, 2, 3)):
            return "no_load"

    return "statistical_outlier"


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

    grp: list[list[pd.Series]] = []
    cur: list[pd.Series] = []
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

        if SELECTED_UNIT is None:
            ev_type = classify_fault_multi(seg)
        else:
            ev_type = classify_fault_single(seg)

        events.append({
            "start": start,
            "end": end,
            "duration_min": int(duration.total_seconds() // 60),
            "n_points": len(seg),
            "median_value": float(med),
            "min_anomaly_score": float(min_score) if pd.notna(min_score) else np.nan,
            "type": ev_type,
        })
    print(x)
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

def run_all_detectors_and_compare(prepared: pd.DataFrame) -> pd.DataFrame:
    
    base = prepared.copy()
    all_results = base.copy()

    for method in COMPARISON_METHODS:
        print(f"[INFO] Uruchamiam detektor={method} (tryb porównawczy)")
        d_method = apply_detector(base, method)

        score_col = f"{method}_score"
        label_col = f"{method}_label"

        if ANOMALY_SCORE_COL in d_method.columns:
            all_results[score_col] = d_method[ANOMALY_SCORE_COL].values
        if ANOMALY_LABEL_COL in d_method.columns:
            all_results[label_col] = d_method[ANOMALY_LABEL_COL].values

        save_outputs(d_method, method)

   
    print("\n[INFO] Korelacja Spearmana anomaly_score pomiędzy metodami:")
    for i, m1 in enumerate(COMPARISON_METHODS):
        for m2 in COMPARISON_METHODS[i + 1:]:
            c1 = all_results.get(f"{m1}_score")
            c2 = all_results.get(f"{m2}_score")
            if c1 is None or c2 is None:
                continue
            rho = _spearman_corr(c1, c2)
            print(f"  {m1} vs {m2}: {rho:.4f}")

    
    unit = _unit_from_df(all_results)
    ppe = _ppe_from_df(all_results)
    out_csv = OUT_DIR / f"comparison_scores_{unit}_{ppe}.csv"
    all_results.to_csv(out_csv, index=False)
    print(f"[OK] Zapisano zbiorcze wyniki porównania do: {out_csv}")
    report = build_comparison_report(all_results)
    heatmap_path = OUT_DIR / f"comparison_heatmap_{unit}_{ppe}.png"
    plot_comparison_heatmap(report, heatmap_path)
    print("\n[INFO] Podsumowanie porównania metod:")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(report.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))

    unit = _unit_from_df(all_results)
    ppe  = _ppe_from_df(all_results)
    summary_path = OUT_DIR / f"comparison_summary_{unit}_{ppe}.csv"
    report.to_csv(summary_path, index=False)
    print(f"[OK] Zapisano podsumowanie porównania do: {summary_path}")
    return all_results

def plot_comparison_heatmap(report, out_path):
   
    methods = sorted(set(report["method_1"]).union(report["method_2"]))

    
    n = len(methods)
    spearman = np.zeros((n, n))
    agreement = np.zeros((n, n))

    
    for _, row in report.iterrows():
        i = methods.index(row["method_1"])
        j = methods.index(row["method_2"])

        spearman[i, j] = spearman[j, i] = row["spearman_score"]
        agreement[i, j] = agreement[j, i] = row["agreement_pct"]

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))

    titles = ["Spearman (ranking score)", "Agreement (%)"]
    matrices = [spearman, agreement]

    for ax, title, mat in zip(axs, titles, matrices):
        im = ax.imshow(mat, cmap="viridis")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(methods)
        ax.set_yticklabels(methods)
        ax.set_title(title)

        
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                if "Spearman" in title:
                    text = f"{val:.2f}"     
                else:
                    text = f"{val:.1f}%"    
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[OK] Zapisano heatmapę porównania do: {out_path}")

def run_pipeline(method: str):
    df = load_long()
    df = filter_by_unit(df)
    d = pick_one_ppe(df)
    d = validate_selection(d)

    if SELECTED_UNIT is None:
        print("[INFO] SELECTED_UNIT=None -> uruchamiam reshape_three_phase (tryb trójfazowy)")
        d = reshape_three_phase(d)
    else:
        print(f"[INFO] SELECTED_UNIT={SELECTED_UNIT} -> pomijam reshape_three_phase (tryb jednowymiarowy)")

    d = add_features(d)
    d = enrich_physics(d)

    if method == "compare":
        run_all_detectors_and_compare(d)
    else:
        d = apply_detector(d, method)
        save_outputs(d, method)
    


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anomaly detection pipeline")
    parser.add_argument(
        "--method",
        choices=sorted(list(DETECTORS.keys()) + ["compare"]),
        default=DETECTION_METHOD,
        help="Wybierz metodę detekcji anomalii (lub 'compare' aby uruchomić wszystkie i porównać)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    method = args.method or DETECTION_METHOD
    run_pipeline(method)


if __name__ == "__main__":
    main()
