"""Export daily simulated flow per flowpath from the NeuralNGEN t-route runs (input to `drought`).

Reads <gauges_root>/gage-<id>/outputs/troute_<scenario>_stream/troute_output_*.nc (variables
flow (feature_id, time), 5-minute steps, m3/s), averages to daily means and writes
<out_dir>/<id>.parquet with index date and one column per flowpath id.

python -m lowflow.troute_flow --config configs/default.yml --scenario all --out_dir data/sim_flow/neuralngen_all
"""
import argparse
import glob
from pathlib import Path

import pandas as pd
import xarray as xr

from .config import load_config, read_basins


def basin_flow(gauges_root, gid, scenario) -> pd.DataFrame:
    files = sorted(glob.glob(f"{gauges_root}/gage-{gid}/outputs/troute_{scenario}_stream/troute_output_*.nc"))
    if not files:
        return None
    parts = []
    for f in files:
        with xr.open_dataset(f) as ds:
            q = ds["flow"].transpose("time", "feature_id").to_pandas()
        q.index = pd.DatetimeIndex(q.index) - pd.Timedelta(minutes=5)      # interval-ending stamps
        parts.append(q)
    q = pd.concat(parts).sort_index()
    q = q[~q.index.duplicated()]
    d = q.resample("D").mean()
    d.columns = [int(c) for c in d.columns]
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scenario", default="all", help="mse | fdc | rfl | all")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for gid in read_basins(cfg):
        d = basin_flow(cfg.gauges_root, gid, a.scenario)
        if d is None:
            print(f"{gid}: no t-route output"); continue
        d.columns = d.columns.astype(str)
        d.to_parquet(out / f"{gid}.parquet")
        print(f"{gid}: {d.shape[1]} flowpaths, {d.index.min().date()} to {d.index.max().date()}", flush=True)


if __name__ == "__main__":
    main()
