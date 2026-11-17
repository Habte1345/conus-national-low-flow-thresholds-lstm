"""
Compute observed low-flow thresholds from USGS daily streamflow and write the LSTM target table.

Percentiles: P5, P10, P15, P20, P25, P30  (Weibull plotting position, non-exceedance)
  fixed    (ft): one value per site from the full daily record
  variable (vt): one value per site and calendar day-of-year (1..366), pooling all days within
                 ±WINDOW days of that DOY across all years (circular), then a circular
                 SMOOTH_DAYS moving average of the resulting curve
  frequency(fq): taken as-is from FREQ_FILE (LP3 dQT, m3/s) if it exists; otherwise skipped

Streamflow source: FLOW_FILE — wide parquet, index = date, one column per site_no, values in m3/s
  (F:/HPC/conus-lowflow-hydro-drought-thresholds/data/usgs_observations/usgs_daily_flow_ref.parquet)
Drainage area (km2) for the mm/d conversion: AREA_FILE (GAGES-II, DRAIN_SQKM)
Sites: every column of FLOW_FILE that has an area and >= 10 years of data. data/basin_list.txt is NOT touched.

Usage:  python src/make_thresholds.py [--flow_file PATH] [--window 15] [--smooth 7]

Outputs (data/):
  fixed_thresholds.parquet     site_no, threshold, value_mmd, value_m3s
  variable_thresholds.parquet  site_no, doy, threshold, value_mmd, value_m3s
  thresholds.parquet           site_no, family, name, doy, value[mm/d]   <- LSTM target table
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PERCENTILES = [5, 10, 15, 20, 25, 30]
NAMES = [f"P{p}" for p in PERCENTILES]
FREQ_FILE = Path("/scratch/hdagne1/conus-lowflow-hydro-drought-thresholds/thresholds/low-flow-metrics/frequency_thresholds.parquet")
AREA_FILE = Path("/scratch/hdagne1/conus-lowflow-hydro-drought-thresholds/data/gage-II/gages_ii.parquet")  # DRAIN_SQKM
FLOW_FILE = Path("/scratch/hdagne1/conus-lowflow-hydro-drought-thresholds/data/usgs_observations/usgs_daily_flow_ref.parquet")

# ---------------------------------------------------------------------------------------
def weibull_percentiles(x: np.ndarray, pcts=PERCENTILES) -> np.ndarray:
    """Non-exceedance percentiles by Weibull plotting position p_i = i/(n+1), linear interpolation."""
    x = np.sort(x[np.isfinite(x)])
    n = len(x)
    if n == 0:
        return np.full(len(pcts), np.nan)
    p = np.arange(1, n + 1) / (n + 1)
    return np.interp(np.asarray(pcts) / 100.0, p, x)


def load_flow(flow_file: Path, area_file: Path):
    """Wide daily table (m3/s) -> dict site -> DataFrame(mmd, m3s); needs DRAIN_SQKM for every site."""
    q = pd.read_parquet(flow_file)
    q.index = pd.to_datetime(q.index)
    q.columns = [str(c).zfill(8) for c in q.columns]
    g = pd.read_parquet(area_file)
    g["site_no"] = g["site_no"].astype(str).str.zfill(8)
    area = g.drop_duplicates("site_no").set_index("site_no")["DRAIN_SQKM"].astype(float)
    return q, area


def site_frame(q: pd.DataFrame, area: pd.Series, site: str) -> pd.DataFrame:
    m3s = q[site].astype(float)
    m3s[m3s < 0] = np.nan
    mmd = m3s * 86.4 / area[site]                          # m3/s -> mm/d
    return pd.DataFrame({"mmd": mmd, "m3s": m3s}).dropna(how="all")


def variable_thresholds(q: pd.DataFrame, window: int, smooth: int) -> pd.DataFrame:
    """[366, n_pct] table per unit: pooled ±window-day Weibull percentiles, circular smoothing."""
    doy = q.index.dayofyear.values
    out = {}
    for unit in ["mmd", "m3s"]:
        v = q[unit].values
        ok = np.isfinite(v)
        tab = np.full((366, len(PERCENTILES)), np.nan)
        for d in range(1, 367):
            dist = np.abs(doy - d)
            dist = np.minimum(dist, 366 - dist)            # circular distance
            sel = ok & (dist <= window)
            tab[d - 1] = weibull_percentiles(v[sel])
        if smooth and smooth > 1:                          # circular moving average of the curve
            k = smooth // 2
            ext = np.concatenate([tab[-k:], tab, tab[:k]], axis=0)
            tab = np.array([np.nanmean(ext[i:i + smooth], axis=0) for i in range(366)])
        out[unit] = tab
    return out


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow_file", default=str(FLOW_FILE))
    ap.add_argument("--area_file", default=str(AREA_FILE))
    ap.add_argument("--window", type=int, default=15, help="±days pooled per DOY (RO3: 15-30)")
    ap.add_argument("--smooth", type=int, default=7, help="circular moving-average length in days (1 = off)")
    a = ap.parse_args()

    qall, area = load_flow(Path(a.flow_file), Path(a.area_file))
    sites = [s for s in qall.columns if s in area.index]
    no_area = [s for s in qall.columns if s not in area.index]
    print(f"{len(sites)} sites with area ({len(no_area)} without) · window ±{a.window} d · "
          f"smooth {a.smooth} d · percentiles {NAMES}")

    ft_rows, vt_rows, skipped = [], [], list(no_area)
    for i, site in enumerate(sites, 1):
        q = site_frame(qall, area, site)
        nyears = q["mmd"].notna().sum() / 365.25
        if nyears < 10:
            skipped.append(site); continue

        f_mmd = weibull_percentiles(q["mmd"].values)
        f_m3s = weibull_percentiles(q["m3s"].values)
        ft_rows += [(site, n, f_mmd[k], f_m3s[k]) for k, n in enumerate(NAMES)]

        vt = variable_thresholds(q, a.window, a.smooth)
        for d in range(366):
            vt_rows += [(site, d + 1, n, vt["mmd"][d, k], vt["m3s"][d, k]) for k, n in enumerate(NAMES)]

        if i % 25 == 0 or i == len(sites):
            print(f"  {i}/{len(sites)}  {site}  {nyears:.1f} yr  P10={f_m3s[1]:.3f} m3/s")

    ft = pd.DataFrame(ft_rows, columns=["site_no", "threshold", "value_mmd", "value_m3s"])
    vt = pd.DataFrame(vt_rows, columns=["site_no", "doy", "threshold", "value_mmd", "value_m3s"])
    Path("data").mkdir(exist_ok=True)
    ft.to_parquet("data/fixed_thresholds.parquet", index=False)
    vt.to_parquet("data/variable_thresholds.parquet", index=False)

    # ---- LSTM target table -----------------------------------------------------------------
    parts = [
        ft.assign(family="ft", doy=np.nan).rename(columns={"threshold": "name", "value_mmd": "value"}),
        vt.assign(family="vt").rename(columns={"threshold": "name", "value_mmd": "value"}),
    ]
    if FREQ_FILE.exists():
        fq = pd.read_parquet(FREQ_FILE)
        fq["site_no"] = fq["site_no"].astype(str).str.zfill(8)
        fq = fq[fq["site_no"].isin(ft["site_no"].unique())]
        fq = fq.melt(id_vars="site_no", var_name="name", value_name="value_m3s")
        fq["name"] = fq["name"].str.replace("_m3s", "", regex=False).str.replace(r"^(\d+)Q(\d+)$", r"Q\1_\2", regex=True)
        fq["value"] = fq["value_m3s"] * 86.4 / fq["site_no"].map(area)
        parts.append(fq.assign(family="fq", doy=np.nan))
        print(f"frequency thresholds included from {FREQ_FILE.name}")
    else:
        print("frequency thresholds NOT included (FREQ_FILE not found)")

    out = pd.concat(parts, ignore_index=True)[["site_no", "family", "name", "doy", "value"]]
    out["site_no"] = out["site_no"].astype(str).str.zfill(8)
    out.to_parquet("data/thresholds.parquet", index=False)

    # ---- report ------------------------------------------------------------------------------
    print("\nrows per family:"); print(out.groupby("family").size().to_string())
    print("names:", sorted(out["name"].unique()))
    print("sites:", out["site_no"].nunique(), "| NaN:", int(out["value"].isna().sum()),
          "| negative:", int((out["value"] < 0).sum()))
    if skipped:
        print(f"skipped {len(skipped)} sites (no DRAIN_SQKM or <10 yr of data): {skipped[:10]} ...")
    piv = ft.pivot(index="site_no", columns="threshold", values="value_mmd")[NAMES]
    print("P5<P10<…<P30 at every site:", bool((piv.diff(axis=1).iloc[:, 1:] >= 0).all().all()))
    print("wrote data/fixed_thresholds.parquet, data/variable_thresholds.parquet, data/thresholds.parquet")


if __name__ == "__main__":
    main()