"""Build the per-basin cache used by training and prediction.

For every gauged basin (outlet gauge of an NGIAB subset):
  1. read divides, attributes and the flowpath network from the subset geopackage
  2. aggregate hourly AORC forcing to daily values per divide
  3. for every flowpath f, form the area weights A_c / A_f over its upstream divides U(f),
     the area-weighted static attributes, and the drainage area A_f
  4. compute observed thresholds at the outlet and at every interior USGS gauge of the basin,
     with flow converted to depth over the drainage area of the gauge flowpath
  5. store the upstream-aggregated daily forcing of each gauge flowpath (training inputs)

Output: <cache_dir>/basins/<gauge>.npz and <cache_dir>/targets.csv
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .config import load_config, read_basins
from .forcing import daily_divide_forcing, forcing_path
from .hydrofabric import Basin, aggregate_static
from .thresholds import gauge_thresholds


def read_outlet_flow(cfg, gid) -> pd.Series:
    f = Path(cfg.streamflow_dir) / f"{gid}.parquet"
    df = pd.read_parquet(f)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index("datetime")
    idx = pd.to_datetime(df.index)
    idx = idx.tz_localize(None) if idx.tz is not None else idx
    s = pd.Series(df["flow_m3s"].values, index=idx).sort_index()
    return s.groupby(s.index.normalize()).mean()


def read_interior_flow(cfg, gid) -> pd.Series:
    f = Path(cfg.interior_obs_dir) / f"{gid}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f, parse_dates=["datetime"])
    s = pd.Series(df["Q_m3s"].values, index=pd.DatetimeIndex(df["datetime"]).tz_localize(None)).sort_index()
    return s.groupby(s.index.normalize()).mean()


def interior_table(cfg) -> pd.DataFrame:
    if not cfg.get("interior_file") or not Path(cfg.interior_file).exists():
        return pd.DataFrame(columns=["outlet_gage", "interior_gage", "flowpath"])
    t = pd.read_csv(cfg.interior_file, dtype=str)
    t["outlet_gage"] = t.outlet_gage.str.zfill(8)
    t["interior_gage"] = t.interior_gage.str.zfill(8)
    t["flowpath"] = t.flowpath.astype(float).astype(int)
    if cfg.get("interior_exclude_file") and Path(cfg.interior_exclude_file).exists():
        drop = {l.strip().zfill(8) for l in Path(cfg.interior_exclude_file).read_text().split() if l.strip()}
        t = t[~t.interior_gage.isin(drop)]
    return t.drop_duplicates("interior_gage")


def build_basin(cfg, gid, interior, out_dir) -> list:
    b = Basin(cfg.gauges_root, gid, cfg.static_attributes)
    dates, X = daily_divide_forcing(forcing_path(cfg.gauges_root, gid), b.divide_n, cfg.dynamic_inputs, cfg.precip_var)

    fps = b.fp_n
    rows, cols, vals, area = b.weights(fps)
    W = sp.csr_matrix((vals, (rows, cols)), shape=(len(fps), len(b.divide_n)))
    fstat = aggregate_static(b.static, b.static_names, rows, cols, vals, len(fps))
    fpos = {f: i for i, f in enumerate(fps)}

    # gauges of this basin: outlet + interior gauges on flowpaths of this subset
    gauges = [("outlet", gid, b.outlet(), read_outlet_flow(cfg, gid))]
    for r in interior[interior.outlet_gage == gid].itertuples():
        if r.flowpath in fpos and r.interior_gage != gid:
            gauges.append(("interior", r.interior_gage, int(r.flowpath), read_interior_flow(cfg, r.interior_gage)))

    recs, tvt, tfq, tX, tfp = [], [], [], [], []
    Xd = X.reshape(len(b.divide_n), -1)
    for kind, g, f, q in gauges:
        if q is None:
            continue
        A = float(area[fpos[f]])
        th = gauge_thresholds(q * 86.4 / A, cfg)                          # m3/s -> mm/d over A_f
        if th is None:
            continue
        recs.append(dict(basin=gid, gauge=g, kind=kind, flowpath=int(f), area_km2=A, years=th["years"]))
        tvt.append(th["vt"]); tfq.append(th["fq"]); tfp.append(f)
        tX.append((W[fpos[f]] @ Xd).reshape(X.shape[1], X.shape[2]).astype(np.float32))

    np.savez_compressed(
        out_dir / f"{gid}.npz",
        dates=dates.values.astype("datetime64[D]").astype(np.int64),
        divide_n=b.divide_n, divide_forcing=X,
        fp=fps, fp_area=area, fp_static=fstat.astype(np.float32),
        W_data=W.data, W_indices=W.indices, W_indptr=W.indptr, W_shape=np.array(W.shape),
        target_fp=np.array(tfp, dtype=np.int64),
        target_vt=np.array(tvt, dtype=np.float32).reshape(len(tfp), 366, len(cfg.percentiles)),
        target_fq=np.array(tfq, dtype=np.float32).reshape(len(tfp), len(cfg.nqy)),
        target_forcing=np.array(tX, dtype=np.float32).reshape(len(tfp), X.shape[1], X.shape[2]),
        static_names=np.array(b.static_names),
        target_meta=np.array(json.dumps(recs)),
    )
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--basins", nargs="*", help="subset of outlet gauges (default: basin_file)")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    out_dir = cfg.cache_dir / "basins"
    out_dir.mkdir(parents=True, exist_ok=True)
    interior = interior_table(cfg)
    basins = a.basins or read_basins(cfg)

    recs, failed = [], []
    for k, gid in enumerate(basins, 1):
        gid = gid.zfill(8)
        try:
            if (out_dir / f"{gid}.npz").exists() and not a.overwrite:
                recs += json.loads(str(np.load(out_dir / f"{gid}.npz")["target_meta"]))
                continue
            recs += build_basin(cfg, gid, interior, out_dir)
            print(f"[{k}/{len(basins)}] {gid} done", flush=True)
        except Exception as e:                                             # keep going, report at the end
            failed.append(gid)
            print(f"[{k}/{len(basins)}] {gid} FAILED: {e}", flush=True)
            traceback.print_exc(file=sys.stdout)
    t = pd.DataFrame(recs)
    t.to_csv(cfg.cache_dir / "targets.csv", index=False)
    print(f"targets: {len(t)} gauges ({(t.kind == 'outlet').sum() if len(t) else 0} outlets, "
          f"{(t.kind == 'interior').sum() if len(t) else 0} interior) in {t.basin.nunique() if len(t) else 0} basins")
    if failed:
        print(f"failed basins ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
