
from typing import Iterable, List, Tuple
import re
from pathlib import Path
import pandas as pd
import load_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_MID_DIR = PROJECT_ROOT / "data" / "output_mid"
OUT_DIR = PROJECT_ROOT / "data" / "output"
OUT_MID_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

META_COLS_CANON = [
    "ilosc_oczekiwanych_wartosci",
    "ilosc_brakujacych_wartosci",
    "ilosc_blednych_wartosci",
    "dane_za_dobe",
    "numer_seryjny",
    "typ_urzadzenia",
    "numer_ppe",
    "obis_skrot",
    "obis_pelny",
    "system_odczytowy",
    "wielkosc_mierzona",
    "typ_wielkosci_mierzonej",
    "rodzaj_wartosci",
    "interwal_integracji_usredniania",
    "jednostka",
    "status_wiersza",
    "mnozna_wiersza",
]


_TIME_COL_REGEX = re.compile(r"^(?:[01]?\d|2[0-3]):[0-5]\d$")
_IS_2400 = re.compile(r"^24:00$")


def split_meta_and_time_cols(columns: Iterable[str]) -> Tuple[List[str], List[str], List[str]]:
    cols = list(columns)
    meta = [c for c in cols if c in META_COLS_CANON]
    
    time_cols = [c for c in cols if _TIME_COL_REGEX.match(c)]
    
    if any(_IS_2400.match(c) for c in cols):
        time_cols.append("24:00")
   
    other = [c for c in cols if c not in meta and c not in time_cols]
    return meta, time_cols, other


def to_long(df: pd.DataFrame, *, date_col: str = "dane_za_dobe", value_col: str = "wartosc") -> pd.DataFrame:
    meta, time_cols, _ = split_meta_and_time_cols(df.columns)

    long_df = df.melt(id_vars=meta, value_vars=time_cols, var_name="godzina_hhmm", value_name=value_col)

    
    def _pad_hhmm(s: str) -> str:
        hh, mm = str(s).split(":")
        return f"{int(hh):02d}:{int(mm):02d}"

    long_df["godzina_hhmm"] = long_df["godzina_hhmm"].map(_pad_hhmm)

    
    date_parsed = pd.to_datetime(long_df[date_col], errors="coerce", dayfirst=True)

    
    is_2400 = long_df["godzina_hhmm"] == "24:00"
    hhmm = long_df["godzina_hhmm"].where(~is_2400, "00:00")
    date_adjusted = date_parsed.where(~is_2400, date_parsed + pd.Timedelta(days=1))

    long_df["timestamp"] = pd.to_datetime(
        date_adjusted.dt.strftime("%Y-%m-%d") + " " + hhmm, errors="coerce"
    )

    
    long_df[value_col] = pd.to_numeric(long_df[value_col], errors="coerce")

   
    first_cols = ["timestamp", value_col]
    rest_cols = [c for c in long_df.columns if c not in first_cols]
    long_df = long_df[first_cols + rest_cols]

    
    if long_df["timestamp"].isna().any():
        n = long_df["timestamp"].isna().sum()
        print(f"[WARN] {n} wierszy ma niepoprawny timestamp (zostaną zachowane, ale sprawdź źródło).")

    return long_df


def save_long(df_long: pd.DataFrame, *, basename: str = "data_long"):
    
    
    pqt_all = OUT_MID_DIR / f"{basename}.parquet"
    csv_all = OUT_MID_DIR / f"{basename}.csv"
    df_long.to_parquet(pqt_all, index=False)
    
    df_long.to_csv(csv_all, index=False)

   
    if "numer_ppe" in df_long.columns:
        per_ppe_dir = OUT_MID_DIR / "per_ppe"
        per_ppe_dir.mkdir(exist_ok=True)
        for ppe, g in df_long.groupby("numer_ppe", sort=False):
            safe_ppe = str(ppe).replace("/", "_")
            g.to_parquet(per_ppe_dir / f"ppe_{safe_ppe}.parquet", index=False)

    print(f"[OK] Zapisano: {pqt_all}")
    print(f"[OK] Zapisano: {csv_all}")
    if "numer_ppe" in df_long.columns:
        print(f"[OK] Zapisano partycje per_ppe/ (Parquet).")


if __name__ == "__main__":
    df_long = to_long(load_data.df)
    save_long(df_long, basename="data_long")
