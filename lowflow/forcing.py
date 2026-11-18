"""Hourly NGIAB AORC forcing (forcings/forcings.nc) -> daily divide forcing.

The ngiab_data_cli file stores divides along 'catchment-id' with identifiers in the 'ids' data
variable and the clock in 'Time' (epoch seconds, one row per divide). Precipitation is converted
to a daily total in mm/d, every other variable to a daily mean. The file is read in divide chunks,
so the hourly array is never held in memory at once.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .hydrofabric import num


def _clock(ds) -> pd.DatetimeIndex:
    if "time" in ds.coords:
        return pd.DatetimeIndex(pd.to_datetime(ds["time"].values))
    t = ds["Time"].values
    t = t[0] if t.ndim == 2 else t
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(t), unit="s", utc=True)).tz_localize(None)


def _ids(ds) -> list:
    for k in ("ids", "catchment-id", "divide_id"):
        if k in ds.variables:
            return [str(v) for v in ds[k].values]
    raise KeyError("forcings.nc has no 'ids', 'catchment-id' or 'divide_id'")


def _precip_factor(units: str) -> tuple:
    """(daily reducer, factor) for precipitation from its units attribute."""
    u = (units or "").replace(" ", "").lower()
    if any(s in u for s in ("/s", "s-1", "s^-1", "s**-1")):          # rate (kg m-2 s-1 == mm/s)
        return "mean", 86400.0
    return "sum", 1.0                                                  # hourly accumulation (mm)


def daily_divide_forcing(nc_path, divide_n, variables, precip_var, chunk=64):
    """Return (dates, X) with X of shape (n_div, n_days, n_var) aligned to divide_n."""
    ds = xr.open_dataset(nc_path)
    try:
        ids = [num(i) for i in _ids(ds)]
        pos = {n: i for i, n in enumerate(ids)}
        missing = [n for n in divide_n if n not in pos]
        if missing:
            raise KeyError(f"{len(missing)} divides of the gpkg are absent from {nc_path}: {missing[:5]}")
        hours = _clock(ds)
        days = hours.normalize()
        dates = pd.DatetimeIndex(np.unique(days))
        start = np.searchsorted(days.values, dates.values)               # first hour of every day
        nh = np.diff(np.r_[start, len(days)])
        full = nh == 24                                                  # incomplete days are dropped
        dims = ds[variables[0]].dims
        cdim = next(d for d in dims if ds.sizes[d] == len(ids) and ds.sizes[d] != len(hours))
        tdim = next(d for d in dims if d != cdim)
        X = np.full((len(divide_n), len(dates), len(variables)), np.nan, np.float32)
        rows = np.array([pos[n] for n in divide_n])
        for v, var in enumerate(variables):
            red, fac = ("mean", 1.0)
            if var == precip_var:
                red, fac = _precip_factor(ds[var].attrs.get("units", ""))
            for i in range(0, len(rows), chunk):
                r = rows[i:i + chunk]
                o = np.argsort(r)                                        # netCDF reads need increasing indices
                a = ds[var].isel({cdim: r[o]}).transpose(cdim, tdim).values.astype(np.float64)
                a = a[np.argsort(o)]                                     # (chunk, n_hours), back in divide order
                s = np.add.reduceat(a, start, axis=1)
                d = (s / nh) if red == "mean" else s
                X[i:i + chunk, :, v] = (d * fac).astype(np.float32)
        X[:, ~full, :] = np.nan
        return dates, X
    finally:
        ds.close()


def forcing_path(gauges_root, gauge_id) -> Path:
    return Path(gauges_root) / f"gage-{gauge_id}" / "forcings" / "forcings.nc"
