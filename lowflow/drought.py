"""Streamflow drought at every flowpath from simulated flow and the learned thresholds.

Simulated daily flow (t-route output of the NeuralNGEN runs, or NWM retrospective) is supplied
per basin as a wide parquet (index date, one column per flowpath id, m3/s) and converted to depth
over the drainage area of each flowpath.

Modes (cfg key or --mode)
  mapped   default. Day-of-year percentiles of the simulated flow, T_sim,k(d), are mapped onto
           the learned thresholds, T_k(d), by a monotone piecewise-linear transform in log space
           through the anchors (log T_sim,k, log T_k), with a constant shift below the lowest
           and above the highest anchor. Drought timing then follows the simulated percentiles
           (bias in the simulation cancels), while deficit volumes are expressed in units
           anchored to the observed thresholds.
  model    simulated flow against its own day-of-year percentiles (fully model-consistent).
  learned  simulated flow against the learned thresholds without mapping (mixes two sources;
           kept only for comparison).

Drought classes (threshold bands)  none Q >= P30, D0 P25-P30, D1 P20-P25, D2 P10-P20, D3 P5-P10,
D4 Q < P5. Events are runs below a threshold with gaps of up to pool_days merged and at least
min_days long. Per flowpath and threshold: events per year, mean and maximum duration (d),
drought days per year, deficit volume per year (mm), median onset day of year.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .hydrofabric import num
from .thresholds import doy_percentiles

CLASS_BANDS = [("D4", "P5"), ("D3", "P10"), ("D2", "P20"), ("D1", "P25"), ("D0", "P30")]


def quantile_map(x_log, s_anchor, l_anchor):
    """Monotone piecewise-linear map of x (log space) through anchors s -> l, shifts outside."""
    s = np.maximum.accumulate(s_anchor + np.arange(len(s_anchor)) * 1e-9, axis=-1)
    y = np.interp(x_log, s, l_anchor)
    y = np.where(x_log < s[0], l_anchor[0] + (x_log - s[0]), y)
    return np.where(x_log > s[-1], l_anchor[-1] + (x_log - s[-1]), y)


def events(below, deficit, pool_days, min_days):
    """Start, end (exclusive) and deficit of pooled drought events."""
    b = below.astype(int)
    edges = np.flatnonzero(np.diff(np.r_[0, b, 0]))
    st, en = list(edges[::2]), list(edges[1::2])
    i = 0
    while i < len(st) - 1:                                                # pool short interruptions
        if st[i + 1] - en[i] <= pool_days:
            en[i] = en[i + 1]; del st[i + 1]; del en[i + 1]
        else:
            i += 1
    out = [(s, e, float(deficit[s:e].sum())) for s, e in zip(st, en) if e - s >= min_days]
    return out


def flowpath_drought(q, T, names, dates, pool_days, min_days):
    """q (n_days,), T (n_days, P) in mm/d -> list of per-threshold metrics and the class series."""
    years = len(dates) / 365.25
    rows = []
    for k, n in enumerate(names):
        below = np.isfinite(q) & (q < T[:, k])
        deficit = np.where(below, T[:, k] - q, 0.0)
        ev = events(below, deficit, pool_days, min_days)
        dur = np.array([e - s for s, e, _ in ev]) if ev else np.array([])
        rows.append(dict(name=n, events_per_year=len(ev) / years,
                         mean_duration_d=float(dur.mean()) if len(dur) else 0.0,
                         max_duration_d=float(dur.max()) if len(dur) else 0.0,
                         drought_days_per_year=float(below.sum()) / years,
                         deficit_mm_per_year=float(deficit.sum()) / years,
                         median_onset_doy=float(np.median(dates[[s for s, _, _ in ev]].dayofyear)) if ev else np.nan))
    cls = np.zeros(len(q), np.int8)                                        # 0 none, 1 D0 ... 5 D4
    pos = {n: k for k, n in enumerate(names)}
    for level, (lab, thr) in zip([5, 4, 3, 2, 1], CLASS_BANDS):
        if thr in pos:
            m = (cls == 0) & np.isfinite(q) & (q < T[:, pos[thr]])
            cls[m] = level
    cls[~np.isfinite(q)] = -1
    return rows, cls


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--flow_dir", required=True, help="<basin>.parquet, index date, columns flowpath ids, m3/s")
    ap.add_argument("--mode", choices=["mapped", "model", "learned"], default="mapped")
    ap.add_argument("--period", nargs=2, default=None, help="evaluation period (default threshold_period)")
    ap.add_argument("--pool_days", type=int, default=5)
    ap.add_argument("--min_days", type=int, default=7)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--save_classes", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, {"family": "vt", "seed": a.seed})
    period = a.period or cfg.threshold_period
    files = sorted(cfg.runs_dir.glob(f"vt_fold*_seed{cfg.seed}/predictions/*.npz"))
    out = cfg.runs_dir / f"drought_{a.mode}_seed{cfg.seed}"; out.mkdir(parents=True, exist_ok=True)

    for f in files:
        b = f.stem
        ff = Path(a.flow_dir) / f"{b}.parquet"
        if not ff.exists():
            continue
        p = np.load(f)
        names = [str(n) for n in p["names"]]; eps = float(p["log_eps"])
        Q = pd.read_parquet(ff); Q.index = pd.to_datetime(Q.index).normalize()
        Q = Q.groupby(level=0).mean().loc[period[0]:period[1]].asfreq("D")
        Q.columns = [num(c) for c in Q.columns]
        dates = Q.index; doy = dates.dayofyear.values - 1
        rows, classes = [], {}
        for i, fp in enumerate(p["flowpath"]):
            fp = int(fp)
            if fp not in Q.columns or not np.isfinite(p["mu_log"][i]).all():
                continue
            q = Q[fp].values * 86.4 / float(p["area_km2"][i])               # mm/d
            T_learn = np.clip(np.exp(p["mu_log"][i]) - eps, 0, None)        # (366, P)
            if a.mode == "learned":
                qq, T = q, T_learn[doy]
            else:
                T_sim = doy_percentiles(pd.Series(q, index=dates), cfg.percentiles, cfg.doy_window, cfg.doy_smooth)
                if a.mode == "model":
                    qq, T = q, T_sim[doy]
                else:
                    xl = np.log(np.clip(q, 0, None) + eps)
                    sl, ll = np.log(T_sim + eps), np.log(T_learn + eps)
                    y = np.array([quantile_map(xl[t], sl[doy[t]], ll[doy[t]]) for t in range(len(q))])
                    qq, T = np.exp(y) - eps, T_learn[doy]
            r, cls = flowpath_drought(qq, T, names, dates, a.pool_days, a.min_days)
            rows += [dict(basin=b, flowpath=fp, area_km2=float(p["area_km2"][i]), **x) for x in r]
            classes[fp] = cls
        pd.DataFrame(rows).to_csv(out / f"{b}_summary.csv", index=False)
        if a.save_classes and classes:
            pd.DataFrame(classes, index=dates).to_parquet(out / f"{b}_classes.parquet")
        print(f"{b}: {len(classes)} flowpaths", flush=True)


if __name__ == "__main__":
    main()
