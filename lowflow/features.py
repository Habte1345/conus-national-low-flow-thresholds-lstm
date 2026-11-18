"""Model inputs and target transforms shared by training and prediction.

Dynamic inputs per day   upstream-area-weighted AORC forcing (+ sin, cos of day of year for 'vt')
Static inputs            upstream-area-weighted attributes, circ_mean.* as (sin, cos), log10 drainage area
Targets                  z = (log(T + eps) - m) / s, one common (m, s) for all percentiles of 'vt'
                         (keeps P5 <= ... <= P30 monotone after standardization), per target for 'fq'
"""
import numpy as np


def doy_features(dates) -> np.ndarray:
    d = np.asarray(dates.dayofyear, dtype=np.float32)
    w = 2 * np.pi * (d - 1) / 365.25
    return np.stack([np.sin(w), np.cos(w)], axis=-1).astype(np.float32)


def static_features(fp_static, names, area_km2) -> np.ndarray:
    cols = []
    for j, n in enumerate(names):
        x = fp_static[:, j].astype(np.float64)
        if n.startswith("circ_mean"):
            r = np.deg2rad(x)
            cols += [np.sin(r), np.cos(r)]
        else:
            cols.append(x)
    cols.append(np.log10(np.asarray(area_km2, np.float64)))
    return np.stack(cols, axis=1).astype(np.float32)


def static_feature_names(names) -> list:
    out = []
    for n in names:
        out += [f"{n}_sin", f"{n}_cos"] if n.startswith("circ_mean") else [n]
    return out + ["log10_area_km2"]


class Scaler:
    """Mean and standard deviation for dynamic, static and target arrays (fitted on training gauges)."""

    def __init__(self, d=None):
        d = d or {}
        self.__dict__.update({k: np.asarray(v, np.float32) if isinstance(v, (list, np.ndarray)) else v
                              for k, v in d.items()})

    @staticmethod
    def _ms(x, axis):
        m = np.nanmean(x, axis=axis); s = np.nanstd(x, axis=axis)
        return m.astype(np.float32), np.where(s < 1e-8, 1.0, s).astype(np.float32)

    def fit(self, dyn, stat, y_log, family):
        self.dyn_mean, self.dyn_std = self._ms(dyn.reshape(-1, dyn.shape[-1]), 0)
        self.stat_mean, self.stat_std = self._ms(stat, 0)
        if family == "vt":
            m, s = np.nanmean(y_log), np.nanstd(y_log)
            n = y_log.shape[-1]
            self.y_mean, self.y_std = np.full(n, m, np.float32), np.full(n, max(s, 1e-6), np.float32)
        else:
            self.y_mean, self.y_std = self._ms(y_log.reshape(-1, y_log.shape[-1]), 0)
        return self

    def dyn(self, x):
        return np.nan_to_num((x - self.dyn_mean) / self.dyn_std, nan=0.0)

    def stat(self, x):
        return np.nan_to_num((x - self.stat_mean) / self.stat_std, nan=0.0)   # missing attribute -> training mean

    def y(self, y_log):
        return (y_log - self.y_mean) / self.y_std

    def y_inv(self, mu_z, sigma_z):
        return mu_z * self.y_std + self.y_mean, sigma_z * self.y_std

    def state(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
