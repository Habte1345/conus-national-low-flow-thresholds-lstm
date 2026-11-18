"""Observed low-flow thresholds from daily streamflow depth (mm/d).

Day-of-year percentiles (variable thresholds, family 'vt')
    For each day of year d = 1..366, all daily flows within +/- W days of d (circular, all years)
    are pooled and the non-exceedance percentile p is taken with the Weibull plotting position
    p_i = i / (n + 1). The 366-day curve is then smoothed with a circular moving average.

Frequency thresholds (family 'fq')
    nQy is the n-day mean annual minimum flow with a y-year recurrence interval. Annual minima of
    the n-day moving mean are taken per climate year (1 April to 31 March). A log-Pearson type III
    distribution is fitted to the non-zero minima, and zero minima are handled with the conditional
    probability adjustment p' = (p - p0) / (1 - p0), where p = 1/y and p0 is the fraction of zero
    minima; nQy = 0 when p0 >= p.
"""
import numpy as np
import pandas as pd
from scipy import stats


def weibull_percentiles(x, pcts) -> np.ndarray:
    x = np.sort(np.asarray(x, float)[np.isfinite(x)])
    n = len(x)
    if n == 0:
        return np.full(len(pcts), np.nan)
    return np.interp(np.asarray(pcts) / 100.0, np.arange(1, n + 1) / (n + 1), x)


def doy_percentiles(q: pd.Series, pcts, window=15, smooth=7) -> np.ndarray:
    """(366, n_pct) table indexed by day of year - 1."""
    doy = q.index.dayofyear.values
    v = q.values.astype(float)
    ok = np.isfinite(v)
    tab = np.full((366, len(pcts)), np.nan)
    for d in range(1, 367):
        dist = np.abs(doy - d)
        dist = np.minimum(dist, 366 - dist)
        tab[d - 1] = weibull_percentiles(v[ok & (dist <= window)], pcts)
    if smooth and smooth > 1:
        k = smooth // 2
        ext = np.concatenate([tab[-k:], tab, tab[:k]])
        tab = np.array([np.nanmean(ext[i:i + smooth], axis=0) for i in range(366)])
    return tab


def climate_year(idx) -> np.ndarray:
    return idx.year - (idx.month < 4)


def nqy(q: pd.Series, n: int, y: float, min_days_frac=0.9) -> float:
    """n-day, y-year low flow (same units as q)."""
    qn = q.rolling(n, min_periods=n).mean()
    cy = climate_year(q.index)
    g = pd.DataFrame({"qn": qn.values, "ok": q.notna().values, "cy": cy})
    agg = g.groupby("cy").agg(m=("qn", "min"), k=("ok", "sum"), t=("ok", "size"))
    amin = agg.loc[agg.k >= min_days_frac * 365, "m"].dropna().values
    if len(amin) < 5:
        return np.nan
    p = 1.0 / y
    p0 = np.mean(amin <= 0)
    if p0 >= p:
        return 0.0
    lx = np.log10(amin[amin > 0])
    padj = (p - p0) / (1.0 - p0)
    g_ = stats.skew(lx, bias=False) if len(lx) > 2 else 0.0
    return float(10 ** stats.pearson3.ppf(padj, g_, loc=lx.mean(), scale=lx.std(ddof=1)))


def parse_nqy(name: str) -> tuple:
    """'7Q10' -> (7, 10.0)."""
    n, y = name.upper().split("Q")
    return int(n), float(y)


def gauge_thresholds(q_mmd: pd.Series, cfg) -> dict:
    """Observed thresholds of one gauge, or None when the record is too short.
    Returns {'vt': (366, n_pct) array, 'fq': (n_nqy,) array, 'years': float}."""
    t0, t1 = pd.Timestamp(cfg.threshold_period[0]), pd.Timestamp(cfg.threshold_period[1])
    q = q_mmd.loc[t0:t1].asfreq("D")
    q[q < 0] = np.nan
    years = q.notna().sum() / 365.25
    if years < cfg.min_years:
        return None
    vt = doy_percentiles(q, cfg.percentiles, cfg.doy_window, cfg.doy_smooth)
    fq = np.array([nqy(q, *parse_nqy(nm)) for nm in cfg.nqy])
    return {"vt": vt, "fq": fq, "years": float(years)}


def target_names(cfg) -> list:
    return [f"P{p}" for p in cfg.percentiles] if cfg.family == "vt" else list(cfg.nqy)
