# detection/indicators.py
import numpy as np
import pandas as pd

def _has_cols(df: pd.DataFrame, cols) -> bool:
    return all(c in df.columns for c in cols)

def compute_S(df: pd.DataFrame,
              cols_u=("U1","U2","U3"),
              cols_i=("I1","I2","I3")) -> pd.Series:
    if not _has_cols(df, cols_u+cols_i):
        return pd.Series(np.nan, index=df.index)
    S = sum(df[u].astype(float) * df[i].astype(float) for u, i in zip(cols_u, cols_i))
    return S

def compute_pf(df: pd.DataFrame,
               P_pobr_col="P_pobr",
               P_odd_col="P_odd",
               S_col="S") -> pd.Series:
    need = [P_pobr_col, P_odd_col, S_col]
    if not _has_cols(df, need):
        return pd.Series(np.nan, index=df.index)
    num = df[P_pobr_col].astype(float) - df[P_odd_col].astype(float)
    den = df[S_col].astype(float).replace(0, np.nan)
    return num / den

def compute_wsp2(df: pd.DataFrame, cols_u=("U1","U2","U3")) -> pd.Series:
    if not _has_cols(df, cols_u):
        return pd.Series(np.nan, index=df.index)
    U1, U2, U3 = (df[c].astype(float) for c in cols_u)
    return pd.concat([(U1-U2).abs(), (U2-U3).abs(), (U3-U1).abs()], axis=1).max(axis=1)

def compute_wsp3(df: pd.DataFrame, cols_u=("U1","U2","U3")) -> pd.Series:
    if not _has_cols(df, cols_u):
        return pd.Series(np.nan, index=df.index)
    U1, U2, U3 = (df[c].astype(float) for c in cols_u)
    
    x = U1*1.0 + U2*(-0.5) + U3*(-0.5)
    y = U2*(np.sqrt(3)/2) + U3*(-np.sqrt(3)/2)  
    return np.sqrt(x*x + y*y)

def flags_from_indicators(df: pd.DataFrame,
                          pf_col="pf",
                          S_col="S",
                          Pp_col="P_pobr",
                          pf_thr=0.617,
                          wsp2_col="wsp2", wsp2_diag=10.0, wsp2_crit=150.0,
                          wsp3_col="wsp3", wsp3_k_sigma=3.0,
                          is_night_col="is_night", P_odd_col="P_odd") -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    
    meanP = df[Pp_col].dropna().astype(float).mean() if Pp_col in df else np.nan
    if pf_col in df and Pp_col in df and not np.isnan(meanP):
        out["pf_flag"] = (df[pf_col] < pf_thr) & (df[Pp_col] >= 0.1 * meanP)
    else:
        out["pf_flag"] = False

   
    out["wsp2_diag"] = (df[wsp2_col] > wsp2_diag) if (wsp2_col in df) else False
    out["wsp2_crit"] = (df[wsp2_col] > wsp2_crit) if (wsp2_col in df) else False

    
    if wsp3_col in df:
        
        x = df[[wsp3_col]].copy()
        ts_col = "timestamp" if "timestamp" in df.columns else None
        if ts_col:
            x = x.set_index(pd.to_datetime(df[ts_col], errors="coerce"))
        mu = x[wsp3_col].rolling("30D", min_periods=48).mean()
        sd = x[wsp3_col].rolling("30D", min_periods=48).std()
        thr = (mu + wsp3_k_sigma * sd).reindex_like(x[wsp3_col])
        flag = x[wsp3_col] > thr
        
        if ts_col:
            out["wsp3_flag"] = flag.reset_index(drop=True).reindex(out.index, fill_value=False)
        else:
            out["wsp3_flag"] = False
    else:
        out["wsp3_flag"] = False

    
    if (P_odd_col in df) and (is_night_col in df):
        out["night_gen"] = (df[P_odd_col] > 0) & df[is_night_col].astype(bool)
    else:
        out["night_gen"] = False

    return out
