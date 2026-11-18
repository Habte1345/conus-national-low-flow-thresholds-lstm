"""Training samples: windows of upstream-aggregated forcing at gauge flowpaths.

A sample is (gauge flowpath i, window end day e). Inputs are the L = seq_length days ending at e;
outputs are scored on the last K = predict_last days. For 'vt' each of those days carries the
observed day-of-year thresholds of its own calendar day, for 'fq' the fixed nQy thresholds.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .features import Scaler, doy_features, static_features


def load_targets(cfg, basins) -> dict:
    """Stack the cached gauge targets of the given basins."""
    out = {k: [] for k in ("X", "dates", "stat", "vt", "fq", "meta")}
    for b in basins:
        f = Path(cfg.cache_dir) / "basins" / f"{b}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=False)
        if len(z["target_fp"]) == 0:
            continue
        fpos = {int(v): i for i, v in enumerate(z["fp"])}
        idx = np.array([fpos[int(f_)] for f_ in z["target_fp"]])
        names = [str(n) for n in z["static_names"]]
        stat = static_features(z["fp_static"][idx], names, z["fp_area"][idx])
        dates = pd.DatetimeIndex(z["dates"].astype("datetime64[D]"))
        meta = json.loads(str(z["target_meta"]))
        for j in range(len(idx)):
            out["X"].append(z["target_forcing"][j]); out["dates"].append(dates); out["stat"].append(stat[j])
            out["vt"].append(z["target_vt"][j]); out["fq"].append(z["target_fq"][j]); out["meta"].append(meta[j])
    out["stat"] = np.array(out["stat"], np.float32)
    out["vt"] = np.array(out["vt"], np.float32)
    out["fq"] = np.array(out["fq"], np.float32)
    out["meta"] = pd.DataFrame(out["meta"])
    return out


def log_target(y, eps):
    return np.log(np.clip(y, 0, None) + eps)


class WindowSampler:
    def __init__(self, data, cfg, scaler: Scaler = None, seed=0):
        self.cfg, self.d = cfg, data
        self.L, self.K = int(cfg.seq_length), int(cfg.predict_last)
        self.rng = np.random.default_rng(seed)
        self.vt = cfg.family == "vt"
        ylog = log_target(data["vt"] if self.vt else data["fq"], cfg.log_eps)
        if scaler is None:
            scaler = Scaler().fit(np.concatenate([x[::7] for x in data["X"]]), data["stat"], ylog, cfg.family)
        self.scaler = scaler
        self.ylog_z = scaler.y(ylog)                                          # vt (n,366,P) / fq (n,Q)
        self.Xz = [scaler.dyn(x).astype(np.float32) for x in data["X"]]
        self.Sz = scaler.stat(data["stat"]).astype(np.float32)
        self.D = [doy_features(d) for d in data["dates"]]
        self.doy = [d.dayofyear.values - 1 for d in data["dates"]]
        self.ends = []                                                        # valid window ends per gauge
        for x in data["X"]:
            bad = np.cumsum(np.r_[0, np.isnan(x).any(1).astype(int)])
            e = np.arange(self.L - 1, len(x))
            self.ends.append(e[(bad[e + 1] - bad[e + 1 - self.L]) == 0])
        self.ok = np.array([len(e) > 0 for e in self.ends])
        if self.vt:
            self.ok &= np.isfinite(self.ylog_z).all(axis=(1, 2))
        else:
            self.ok &= np.isfinite(self.ylog_z).any(axis=1)
        self.n_dyn = data["X"][0].shape[1] + (2 if self.vt else 0)
        self.n_stat = self.Sz.shape[1]
        self.n_out = self.ylog_z.shape[-1]

    def window(self, i, e):
        x = self.Xz[i][e - self.L + 1:e + 1]
        if self.vt:
            x = np.concatenate([x, self.D[i][e - self.L + 1:e + 1]], axis=1)
        s = np.broadcast_to(self.Sz[i], (self.L, self.n_stat))
        return np.concatenate([x, s], axis=1)

    def target(self, i, e):
        if self.vt:
            return self.ylog_z[i][self.doy[i][e - self.K + 1:e + 1]]           # (K, P)
        return np.broadcast_to(self.ylog_z[i], (self.K, self.n_out))          # (K, Q)

    def batch(self, B):
        ids = self.rng.choice(np.flatnonzero(self.ok), size=B)
        xs, ys = [], []
        for i in ids:
            e = self.rng.choice(self.ends[i])
            xs.append(self.window(i, e)); ys.append(self.target(i, e))
        return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)
