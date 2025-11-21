import pandas as pd
from tabulate import tabulate
import unicodedata
import re
from pathlib import Path

try:
    import chardet
except ImportError:
    chardet = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR_IN = PROJECT_ROOT / "data" / "input"
FILENAME = "dane_10_minutowe_posrednie 20250319_192642.csv"

def normalize_col(name: str) -> str:
    name = name.replace("ł", "l").replace("Ł", "L")
    s = ''.join(c for c in unicodedata.normalize('NFKD', name) if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r"[^\w:]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


path = DATA_DIR_IN / FILENAME


def detect_encoding(file_path: Path, *, sample_size: int = 100_000) -> str:
    """
    Heuristically detect encoding of a text file.
    Prefers BOM detection, falls back to chardet if available, then simple trial decodes.
    """
    with file_path.open("rb") as f:
        raw = f.read(sample_size)

    
    bom_map = {
        b"\xef\xbb\xbf": "utf-8-sig",
        b"\xff\xfe": "utf-16",
        b"\xfe\xff": "utf-16",
        b"\xff\xfe\x00\x00": "utf-32",
        b"\x00\x00\xfe\xff": "utf-32",
    }
    for bom, encoding in bom_map.items():
        if raw.startswith(bom):
            return encoding

    if chardet is not None:
        result = chardet.detect(raw)
        encoding = result.get("encoding")
        if encoding:
            return encoding

    fallback_encodings = ("utf-8", "cp1250", "iso-8859-2")
    for encoding in fallback_encodings:
        try:
            raw.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue

    return "utf-8"


encoding = detect_encoding(path)
df = pd.read_csv(
    path,
    sep=";",
    decimal=",",
    encoding=encoding,)


df.columns = [normalize_col(c) for c in df.columns]
# print(f"[INFO] Wykryto kodowanie: {encoding}")
# print(tabulate(df.head(), headers='keys', tablefmt='psql'))
# df["numer_ppe"] = df["numer_ppe"].astype(str)
# print(tabulate(df[["numer_ppe"]].drop_duplicates(), headers='keys', tablefmt='psql'))
ppe = df["numer_ppe"].to_list()