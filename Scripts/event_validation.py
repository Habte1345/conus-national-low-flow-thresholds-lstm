"""
Activity 3.4 — event-based validation of low-flow thresholds against the U.S. Drought Monitor.

Truth: a gauge is "in drought" on a day when >= AREA_MIN % of its HUC8 (or county) is in DLEVEL or worse
       on the USDM map valid for that day (weekly maps, 2000-01-04 onward). USDM events = runs >= USDM_MIN_WEEKS.
Candidates: USGS-FT, USGS-VT, LSTM-FT, LSTM-VT thresholds at P5..P30; streamflow-drought days = flow below
       the threshold; streamflow events = runs >= MIN_DUR days.

Per site, approach and percentile (figures/ev_site_metrics.csv):
  pod        detection rate: share of USDM drought days that are streamflow-drought days
  far        false-alarm rate: share of streamflow-drought days outside USDM drought
  csi        critical success index of drought days
  ev_det     share of USDM events touched by at least one streamflow-drought day
  ev_far     share of streamflow events with no USDM drought day
  onset_lag  first streamflow-event start within [onset-LAG_BEFORE, onset+LAG_AFTER] minus USDM onset (days, median)
  overlap    share of the USDM event's days that are streamflow-drought days (median over events)
  dur_ratio  duration of the matched streamflow event / duration of the USDM event (median over events)

Output: figures/ev_site_metrics.csv (one row per site x approach x percentile). Figures: ev_plots.py.
"""
import glob, sqlite3
from pathlib import Path
import numpy as np, pandas as pd

RUN_FT    = Path("/scratch/hdagne1/conus-national-low-flow-thresholds-lstm/runs/run_2109_1723_ft_seed200")
RUN_VT    = Path("/scratch/hdagne1/conus-national-low-flow-thresholds-lstm/runs/run_2209_0932_vt_seed200")
FLOW_FILE = Path("/scratch/hdagne1/conus-lowflow-hydro-drought-thresholds/data/usgs_observations/usgs_daily_flow_ref.parquet")
AREA_FILE = Path("/scratch/hdagne1/conus-lowflow-hydro-drought-thresholds/data/gage-II/gages_ii.parquet")
USDM_DIR  = Path("/scratch/hdagne1/conus-national-low-flow-thresholds-lstm/data/usdm")
out = Path("/scratch/hdagne1/conus-lowflow-hydro-drought-thresholds/figures")

USDM_AREA = "huc8"          # "huc8" or "fips" (county)
DLEVEL, AREA_MIN = "D1", 50.0
USDM_MIN_WEEKS = 4
MIN_DUR = 7
LAG_BEFORE, LAG_AFTER = 60, 180
NAMES = ["P5", "P10", "P15", "P20", "P25", "P30"]
APPROACHES = ["usgs_ft", "lstm_ft", "usgs_vt", "lstm_vt"]
APP_LABEL = {"usgs_ft": "USGS fixed", "lstm_ft": "LSTM fixed", "usgs_vt": "USGS variable", "lstm_vt": "LSTM variable"}
SNOW_MIN, HUMID_MAX, ARID_MIN, EAST_LON, BFI_MIN = 0.30, 1.0, 1.5, -95.0, 0.60
CLASSES = ["all", "snow", "humid_east", "arid_west", "groundwater"]
SHORT = {"all": "All sites", "snow": "Snow-dominated", "humid_east": "Humid East", "arid_west": "Arid West", "groundwater": "GW-dominated"}

# ---------------------------------------------------------------------------------------
# 1. thresholds (log mm/d) from the two runs
# ---------------------------------------------------------------------------------------
def read_summary(run, family):
    s = pd.read_csv(glob.glob(str(run / f"{family}_*_summary.csv"))[0], dtype={"site_no": str})
    s["site_no"] = s["site_no"].str.zfill(8)
    return s[s["name"].isin(NAMES)].dropna(subset=["obs_log", "mu_log"])

ft = read_summary(RUN_FT, "ft")
vt = read_summary(RUN_VT, "vt"); vt["doy"] = vt["doy"].astype(int)
sites = sorted(set(ft.site_no) & set(vt.site_no))
ft_obs = ft.set_index(["site_no", "name"])["obs_log"]; ft_mu = ft.set_index(["site_no", "name"])["mu_log"]
obs_tab = vt.pivot_table(index=["site_no", "name"], columns="doy", values="obs_log")
mu_tab = vt.pivot_table(index=["site_no", "name"], columns="doy", values="mu_log")
for tab in (obs_tab, mu_tab):
    if 366 not in tab.columns:
        tab[366] = tab[365]
obs_tab, mu_tab = obs_tab[range(1, 367)], mu_tab[range(1, 367)]

# ---------------------------------------------------------------------------------------
# 2. attributes, classes, USDM
# ---------------------------------------------------------------------------------------
g = pd.read_parquet(AREA_FILE); g["site_no"] = g["site_no"].astype(str).str.zfill(8)
g = g.drop_duplicates("site_no").set_index("site_no")
area = g["DRAIN_SQKM"].astype(float)
with sqlite3.connect(RUN_VT / "attributes.db") as c:
    cam = pd.read_sql("SELECT * FROM basin_attributes", c, index_col="gauge_id")
cam.index = cam.index.astype(str).str.zfill(8)


def pick(gcols, gfun, ccol):
    """attribute from GAGES-II when its columns exist, else from the CAMELS attributes.db"""
    if all(c in g.columns for c in gcols):
        return gfun(g).astype(float)
    return cam[ccol].astype(float)

attr = pd.DataFrame(index=g.index.union(cam.index))
attr["frac_snow"] = pick(["SNOW_PCT_PRECIP"], lambda d: d["SNOW_PCT_PRECIP"].astype(float) / 100.0, "frac_snow")
attr["aridity"] = pick(["PET", "PPTAVG_BASIN"], lambda d: d["PET"].astype(float) / (d["PPTAVG_BASIN"].astype(float) * 10.0), "aridity")
attr["bfi"] = pick(["BFI_AVE"], lambda d: d["BFI_AVE"].astype(float) / 100.0, "baseflow_index")
lonc = next((c for c in ["LNG_GAGE", "dec_long_va", "gauge_lon"] if c in g.columns), None)
latc = next((c for c in ["LAT_GAGE", "dec_lat_va", "gauge_lat"] if c in g.columns), None)
attr["lon"] = pd.to_numeric(g[lonc]) if lonc else cam["gauge_lon"].astype(float)
attr["lat"] = pd.to_numeric(g[latc]) if latc else cam["gauge_lat"].astype(float)
adm = pd.read_csv(USDM_DIR / "site_admin.csv", dtype=str).set_index("site_no")
attr = attr.join(adm[["huc8", "fips", "state"]])
sites = [s for s in sites if s in attr.index]
attr = attr.loc[sites]

cls = pd.DataFrame(index=attr.index)
cls["all"] = True
cls["snow"] = attr.frac_snow >= SNOW_MIN
cls["humid_east"] = (attr.aridity <= HUMID_MAX) & (attr.lon >= EAST_LON)
cls["arid_west"] = (attr.aridity >= ARID_MIN) & (attr.lon < EAST_LON)
cls["groundwater"] = attr.bfi >= BFI_MIN
print("sites:", len(sites), "| per class:", cls.sum().to_dict())

usdm_by, usdm_dates = {}, []
for key, fname in [("huc8", "usdm_huc8_weekly.parquet"), ("fips", "usdm_county_weekly.parquet")]:
    f = USDM_DIR / fname
    if not f.exists(): continue
    u = pd.read_parquet(f); u[key] = u[key].astype(str); u["map_date"] = pd.to_datetime(u["map_date"])
    usdm_by.update({(key, k): d.set_index("map_date").sort_index() for k, d in u.groupby(key)})
    usdm_dates.append(u.map_date)
usdm_dates = pd.concat(usdm_dates)
ALT = {"huc8": "fips", "fips": "huc8"}
def usdm_code(site):
    """(kind, code) for the site: preferred area, else the other one; None if neither has USDM data."""
    for kind in (USDM_AREA, ALT[USDM_AREA]):
        code = attr.loc[site, kind]
        if pd.notna(code) and (kind, code) in usdm_by:
            return kind, code
    return None
sites = [s for s in sites if usdm_code(s) is not None]
attr = attr.loc[sites]
print("sites with USDM data:", len(sites), "| via", USDM_AREA, ":", sum(usdm_code(s)[0] == USDM_AREA for s in sites))
cls = cls.loc[sites]

qall = pd.read_parquet(FLOW_FILE); qall.columns = [str(c).zfill(8) for c in qall.columns]
qall.index = pd.to_datetime(qall.index)
t0 = max(usdm_dates.min(), qall.index.min()); t1 = min(usdm_dates.max() + pd.Timedelta(days=6), qall.index.max())
days = pd.date_range(t0, t1, freq="D"); doy_idx = days.dayofyear.values - 1
print(f"evaluation period {t0.date()} .. {t1.date()}  ({len(days)} days)")


def usdm_daily(site):
    """daily USDM percent-area series (D0..D4 cumulative) for the site's area, each map valid for its week."""
    w = usdm_by[usdm_code(site)][["D0", "D1", "D2", "D3", "D4"]]
    return w.reindex(days, method="ffill")


def runs(flag, min_len):
    """[start, end) index pairs of True runs with length >= min_len."""
    idx = np.flatnonzero(np.diff(np.concatenate(([0], flag.astype(int), [0]))))
    st, en = idx[::2], idx[1::2]
    keep = (en - st) >= min_len
    return st[keep], en[keep]


# ---------------------------------------------------------------------------------------
# 3. metrics per site x approach x percentile
# ---------------------------------------------------------------------------------------
def metrics(truth, sf):
    """truth, sf: daily boolean arrays (USDM drought, streamflow drought)."""
    ok = np.isfinite(sf.astype(float))
    hits, miss, fa = (truth & sf).sum(), (truth & ~sf).sum(), (~truth & sf).sum()
    m = dict(pod=hits / max(hits + miss, 1), far=fa / max(hits + fa, 1), csi=hits / max(hits + miss + fa, 1))
    us, ue = runs(truth, USDM_MIN_WEEKS * 7)
    ss, se = runs(sf, MIN_DUR)
    lags, ovl, dur, det = [], [], [], []
    for a, b in zip(us, ue):
        ov = sf[a:b].sum(); ovl.append(ov / (b - a)); det.append(ov > 0)
        cand = ss[(ss >= a - LAG_BEFORE) & (ss <= min(b, a + LAG_AFTER))]
        if len(cand):
            k = np.argmin(np.abs(cand - a)); j = np.flatnonzero(ss == cand[k])[0]
            lags.append(cand[k] - a); dur.append((se[j] - ss[j]) / (b - a))
    m["n_usdm_events"] = len(us)
    m["ev_det"] = np.mean(det) if det else np.nan
    m["ev_far"] = np.mean([(truth[a:b].sum() == 0) for a, b in zip(ss, se)]) if len(ss) else np.nan
    m["onset_lag"] = np.median(lags) if lags else np.nan
    m["overlap"] = np.median(ovl) if ovl else np.nan
    m["dur_ratio"] = np.median(dur) if dur else np.nan
    return m

rows = []
for k, site in enumerate(sites, 1):
    if site not in qall.columns: continue
    q = qall[site].reindex(days).astype(float).to_numpy(copy=True); q[q < 0] = np.nan
    q = q * 86.4 / area[site]
    if np.isfinite(q).sum() < 365 * 5: continue
    truth = (usdm_daily(site)[DLEVEL] >= AREA_MIN).to_numpy() & np.isfinite(q)
    for n in NAMES:
        if (site, n) not in obs_tab.index or (site, n) not in ft_obs.index: continue
        T = {"usgs_ft": np.full(len(q), np.exp(ft_obs[(site, n)])), "lstm_ft": np.full(len(q), np.exp(ft_mu[(site, n)])),
             "usgs_vt": np.exp(obs_tab.loc[(site, n)].to_numpy())[doy_idx], "lstm_vt": np.exp(mu_tab.loc[(site, n)].to_numpy())[doy_idx]}
        for app, t in T.items():
            sf = (q < t) & np.isfinite(q)
            rows.append(dict(site_no=site, name=n, approach=app, usdm_days=int(truth.sum()), **metrics(truth, sf)))
    if k % 50 == 0 or k == len(sites):
        print(f"  {k}/{len(sites)} sites")
res = pd.DataFrame(rows)
out.mkdir(parents=True, exist_ok=True)
res = res.merge(cls.reset_index().rename(columns={"index": "site_no"}), on="site_no", how="left")
res.to_csv(out / "ev_site_metrics.csv", index=False)
print(res.groupby(["approach", "name"])[["pod", "far", "csi", "ev_det", "onset_lag", "overlap"]].median().round(3).to_string())

print(f"wrote {out / 'ev_site_metrics.csv'}  -> run ev_plots.py for the figures")