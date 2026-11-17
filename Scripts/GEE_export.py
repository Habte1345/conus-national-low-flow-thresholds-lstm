"""
Export the threshold products as CSV tables for upload to Google Earth Engine (Assets -> New -> CSV file).
Everything is built from files that already exist; nothing is recomputed.

Tables written to data/gee/:
  gee_gauges.csv       one row per gauge: id, name, lat, lon, huc2/4/6/8, county FIPS, state, area, class flags,
                       attributes (BFI, aridity, snow fraction), fixed thresholds (USGS and LSTM-FT mu/sigma) for
                       P5..P30, frequency thresholds if available, and USDM agreement scores (POD, FAR per family)
  gee_vt_long.csv      one row per gauge x day-of-year: USGS variable and LSTM-VT (mu, sigma) thresholds, P5..P30
  gee_weekly.csv       one row per gauge x USDM week: weekly mean flow (mm/d) and the USDM / LSTM-FT / LSTM-VT class
  gee_huc8_weekly.csv  one row per gauge HUC8 x USDM week: USDM / LSTM-FT / LSTM-VT class as ranks 0..5 (u, f, v)
  gee_national_weekly.csv  per week: % of gauge HUC8s in D1+ and D2+ according to USDM, LSTM-FT, LSTM-VT
  gee_agg_<level>.csv  one row per HUC2 / HUC4 / HUC6 / HUC8 / county / state: gauge count and the median of every
                       threshold and score over the gauges in the unit (what the choropleth layers show)

Upload each CSV as a table asset with lat/lon columns only for gee_gauges.csv (x = lon, y = lat); the others are
plain tables (no geometry). Then set the asset ids in gee_app.js.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd

USDM_DIR  = Path("/bighome/hdagne1/conus-national-low-flow-thresholds-lstm/data/usdm")
RUN_FT    = Path("/bighome/hdagne1/conus-national-low-flow-thresholds-lstm/runs/run_2109_1723_ft_seed200")
RUN_VT    = Path("/bighome/hdagne1/conus-national-low-flow-thresholds-lstm/runs/run_2209_0932_vt_seed200")
RUN_FQ    = None                                   # Path(".../run_xxxx_fq_seed200") once LSTM-FQ is trained
FLOW_FILE = Path("/bighome/hdagne1/conus-lowflow-hydro-drought-thresholds/data/usgs_observations/usgs_daily_flow_ref.parquet")
AREA_FILE = Path("/bighome/hdagne1/conus-lowflow-hydro-drought-thresholds/data/gage-II/gages_ii.parquet")
FIG       = Path("/bighome/hdagne1/conus-lowflow-hydro-drought-thresholds/figures")
OUT       = Path("/bighome/hdagne1/conus-lowflow-hydro-drought-thresholds/data/gee")
NAMES = ["P5", "P10", "P15", "P20", "P25", "P30"]
OUT.mkdir(parents=True, exist_ok=True)

def read_summary(run, family):
    s = pd.read_csv(glob.glob(str(run / f"{family}_*_summary.csv"))[0], dtype={"site_no": str})
    s["site_no"] = s["site_no"].str.zfill(8)
    return s.dropna(subset=["obs_log", "mu_log"])


def pick(df, names, default=np.nan):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(default, index=df.index)


# ---- gauges --------------------------------------------------------------------------------------
g = pd.read_parquet(AREA_FILE); g["site_no"] = g["site_no"].astype(str).str.zfill(8)
g = g.drop_duplicates("site_no").set_index("site_no")
adm = pd.read_csv(USDM_DIR / "site_admin.csv", dtype=str); adm["site_no"] = adm["site_no"].str.zfill(8)
adm = adm.drop_duplicates("site_no").set_index("site_no")
ft = read_summary(RUN_FT, "ft")
vt = read_summary(RUN_VT, "vt"); vt["doy"] = vt["doy"].astype(int)
sites = sorted(set(ft.site_no) & set(vt.site_no) & set(g.index) & set(adm.index))

G = pd.DataFrame(index=pd.Index(sites, name="site_no"))
G["name"] = g.reindex(sites)[next(c for c in ["STANAME", "station_nm", "gauge_name"] if c in g.columns)] if any(c in g.columns for c in ["STANAME", "station_nm", "gauge_name"]) else ""
G["lat"] = pick(g.reindex(sites), ["LAT_GAGE", "dec_lat_va", "gauge_lat"]).values
G["lon"] = pick(g.reindex(sites), ["LNG_GAGE", "dec_long_va", "gauge_lon"]).values
G["area_km2"] = pick(g.reindex(sites), ["DRAIN_SQKM"]).values
G["huc8"] = adm.reindex(sites)["huc8"].str.zfill(8).values
G["huc6"], G["huc4"], G["huc2"] = G.huc8.str[:6], G.huc8.str[:4], G.huc8.str[:2]
G["fips"] = adm.reindex(sites)["fips"].str.zfill(5).values
G["state"] = adm.reindex(sites)["state"].values
G["bfi"] = pick(g.reindex(sites), ["BFI_AVE", "baseflow_index"]).values
G["aridity"] = pick(g.reindex(sites), ["aridity", "ARIDITY"]).values
if G.aridity.isna().all() and {"PET", "PPTAVG_BASIN"} <= set(g.columns):
    G["aridity"] = (g.reindex(sites).PET / (g.reindex(sites).PPTAVG_BASIN * 10)).values      # PPTAVG_BASIN is in cm
G["frac_snow"] = pick(g.reindex(sites), ["frac_snow", "SNOW_PCT_PRECIP"]).values
if G.frac_snow.max() > 1.5:
    G["frac_snow"] = G.frac_snow / 100
G["snow"] = (G.frac_snow >= 0.30).astype(int)
G["humid_east"] = ((G.aridity <= 1.0) & (G.lon >= -95)).astype(int)
G["arid_west"] = ((G.aridity >= 1.5) & (G.lon < -95)).astype(int)
G["groundwater"] = (G.bfi >= 0.60).astype(int)
G["site_id"] = G.index.astype(int)
for lev in ["huc2", "huc4", "huc6", "huc8", "fips", "state"]:
    G[f"{lev}_num"] = G[lev].astype(int)

for PCT in NAMES:                                                 # fixed thresholds, mm/d
    f = ft[ft.name == PCT].set_index("site_no").reindex(sites)
    G[f"usgs_{PCT}"] = np.exp(f.obs_log).values
    G[f"ft_{PCT}_mu"] = np.exp(f.mu_log).values
    G[f"ft_{PCT}_lo"] = np.exp(f.mu_log - f.sigma_log).values    # ~68 % range of the threshold
    G[f"ft_{PCT}_hi"] = np.exp(f.mu_log + f.sigma_log).values
    G[f"ft_{PCT}_crps"] = f.crps_log.values
if RUN_FQ is not None:                                            # frequency thresholds (7Q10 etc.) when available
    fq = read_summary(RUN_FQ, "fq")
    for nm, f in fq.groupby("name"):
        f = f.set_index("site_no").reindex(sites)
        G[f"usgs_{nm}"] = np.exp(f.obs_log).values
        G[f"fq_{nm}_mu"] = np.exp(f.mu_log).values
        G[f"fq_{nm}_lo"] = np.exp(f.mu_log - f.sigma_log).values
        G[f"fq_{nm}_hi"] = np.exp(f.mu_log + f.sigma_log).values

# USDM agreement per gauge, from the all-weeks classes (HUC8 level -> copied to its gauges)
cls = pd.read_csv(FIG / "usdm_all_weeks_classes.csv", dtype={"huc8": str}); cls["huc8"] = cls.huc8.str.zfill(8)
RANK = {c: i for i, c in enumerate(["none", "D0", "D1", "D2", "D3", "D4"])}
ud = cls.USDM.map(RANK) >= 2
for src, tag in [("LSTM-FT", "ft"), ("LSTM-VT", "vt")]:
    ld = cls[src].map(RANK) >= 2
    per = pd.DataFrame({"pod": (ud & ld).groupby(cls.huc8).sum() / ud.groupby(cls.huc8).sum().replace(0, np.nan),
                        "far": (ld & ~ud).groupby(cls.huc8).sum() / ld.groupby(cls.huc8).sum().replace(0, np.nan),
                        "exact": (cls.USDM == cls[src]).groupby(cls.huc8).mean()})
    for k in per.columns:
        G[f"{tag}_{k}"] = (100 * per[k]).reindex(G.huc8).values
G.round(4).reset_index().to_csv(OUT / "gee_gauges.csv", index=False)
print(f"gee_gauges.csv: {len(G)} gauges, {G.shape[1]} columns")

# ---- variable thresholds, long -------------------------------------------------------------------------
V = None
for PCT in NAMES:
    v = vt[vt.name == PCT][["site_no", "doy", "obs_log", "mu_log", "sigma_log"]].set_index(["site_no", "doy"])
    t = pd.DataFrame({f"usgs_{PCT}": np.exp(v.obs_log), f"vt_{PCT}_mu": np.exp(v.mu_log),
                      f"vt_{PCT}_lo": np.exp(v.mu_log - v.sigma_log), f"vt_{PCT}_hi": np.exp(v.mu_log + v.sigma_log)})
    V = t if V is None else V.join(t)
V = V.loc[sites].round(5).reset_index()
V.insert(0, "site_id", V.site_no.astype(int))
V.to_csv(OUT / "gee_vt_long.csv", index=False)
print(f"gee_vt_long.csv: {len(V)} rows")

# ---- weekly flow + classes ----------------------------------------------------------------------------
q = pd.read_parquet(FLOW_FILE); q.columns = [str(c).zfill(8) for c in q.columns]; q.index = pd.to_datetime(q.index)
Q = q[sites].astype(float); Q[Q < 0] = np.nan
Q = Q * 86.4 / G.area_km2.to_numpy()
weeks = pd.to_datetime(sorted(cls.map_date.unique()))
Q7 = Q.rolling(7, min_periods=4).mean().reindex(weeks)
W = Q7.melt(ignore_index=False, var_name="site_no", value_name="q_week").reset_index(names="map_date")
W["huc8"] = W.site_no.map(G.huc8)
W = W.merge(cls.rename(columns={"USDM": "usdm", "LSTM-FT": "ft_class", "LSTM-VT": "vt_class"}).assign(map_date=lambda d: pd.to_datetime(d.map_date)),
            on=["huc8", "map_date"], how="left")
W["ymd"] = W.map_date.dt.strftime("%Y%m%d").astype(int)
W["map_date"] = W.map_date.dt.strftime("%Y-%m-%d")
W.insert(0, "site_id", W.site_no.astype(int))
W.round(5).to_csv(OUT / "gee_weekly.csv", index=False)
print(f"gee_weekly.csv: {len(W)} rows")

# ---- HUC8 weekly classes (USDM / LSTM-FT / LSTM-VT as ranks 0..5) --------------------------------------
H = cls.copy()
H["huc8_num"] = H.huc8.astype(int)
H["ymd"] = pd.to_datetime(H.map_date).dt.strftime("%Y%m%d").astype(int)
for src, col in [("USDM", "u"), ("LSTM-FT", "f"), ("LSTM-VT", "v")]:
    H[col] = H[src].map(RANK).astype(int)
H[["huc8", "huc8_num", "map_date", "ymd", "u", "f", "v"]].to_csv(OUT / "gee_huc8_weekly.csv", index=False)
print(f"gee_huc8_weekly.csv: {len(H)} rows")

# ---- national weekly series: % of gauge HUC8s in drought (D1+) and severe drought (D2+) ----------------------
N = pd.DataFrame({"map_date": sorted(H.map_date.unique())}).set_index("map_date")
for col, name in [("u", "usdm"), ("f", "ft"), ("v", "vt")]:
    N[f"{name}_d1"] = 100 * (H[col] >= 2).groupby(H.map_date).mean()
    N[f"{name}_d2"] = 100 * (H[col] >= 3).groupby(H.map_date).mean()
N = N.reset_index()
N["ymd"] = pd.to_datetime(N.map_date).dt.strftime("%Y%m%d").astype(int)
N.round(2).to_csv(OUT / "gee_national_weekly.csv", index=False)
print(f"gee_national_weekly.csv: {len(N)} weeks")

# ---- aggregates per HUC level / county / state ------------------------------------------------------------
num = [c for c in G.select_dtypes("number").columns if c not in ["lat", "lon", "snow", "humid_east", "arid_west", "groundwater", "site_id"] and not c.endswith("_num")]
for level in ["huc2", "huc4", "huc6", "huc8", "fips", "state"]:
    a = G.groupby(level)[num].median()
    a.insert(0, "n_gauges", G.groupby(level).size())
    for k in ["snow", "humid_east", "arid_west", "groundwater"]:
        a[k] = G.groupby(level)[k].mean()                            # share of gauges in the class
    a = a.round(4).reset_index().rename(columns={level: "code"})
    a.insert(1, "code_num", a.code.astype(int))
    a.to_csv(OUT / f"gee_agg_{level}.csv", index=False)
    print(f"gee_agg_{level}.csv: {len(a)} units")