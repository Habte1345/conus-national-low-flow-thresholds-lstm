"""
Fetch the U.S. Drought Monitor weekly statistics for every gauge (Activity 3.4 truth data).

1. NWIS site service -> HUC8, county FIPS, state for each site           -> data/usdm/site_admin.csv
2. USDM data service, cumulative % area in D0..D4 per week
     by HUC8   -> data/usdm/usdm_huc8_weekly.parquet   (huc8, map_date, D0, D1, D2, D3, D4)
     by county -> data/usdm/usdm_county_weekly.parquet (fips, map_date, D0, D1, D2, D3, D4)
   D1 = percent of the area in D1 *or worse* (traditional / cumulative statistics).

Needs internet. Re-running only fetches what is not cached yet.
If the USDM API is unreachable, download "Comprehensive Statistics" CSVs from droughtmonitor.unl.edu
(area type HUC8 or County, statistics type "Traditional", percent area) and point --usdm_csv at them;
the file is converted to the same parquet layout.

Usage:  python src/usdm_fetch.py [--sites basin_list.txt] [--start 2000-01-01] [--end 2024-12-31]
        (paths are absolute, so it also runs as a notebook cell)
"""
import argparse, io, json, time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode

import numpy as np, pandas as pd

NWIS = "https://waterservices.usgs.gov/nwis/site/"
USDM = "https://usdmdataservices.unl.edu/api"
REPO = Path("/scratch/hdagne1/conus-national-low-flow-thresholds-lstm")
OUT = REPO / "data/usdm"


def get(url, retries=3, sleep=2.0):
    for k in range(retries):
        try:
            with urlopen(Request(url, headers={"User-Agent": "conus-lowflow-thresholds", "Accept": "application/json"}), timeout=60) as r:
                return r.read().decode("utf-8")
        except Exception as e:                     # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(sleep * (k + 1))


# ---------------------------------------------------------------------------------------
# 1. site -> HUC8 / county / state from NWIS
# ---------------------------------------------------------------------------------------
def site_admin(sites):
    f = OUT / "site_admin.csv"
    have = pd.read_csv(f, dtype=str) if f.exists() else pd.DataFrame(columns=["site_no", "huc8", "fips", "state"])
    have["huc8"] = have["huc8"].astype(str).str[:8]
    todo = [s for s in sites if s not in set(have.site_no)]
    rows = []
    for i in range(0, len(todo), 100):
        chunk = todo[i:i + 100]
        q = urlencode({"format": "rdb", "sites": ",".join(chunk), "siteOutput": "expanded", "siteStatus": "all"})
        txt = get(f"{NWIS}?{q}")
        lines = [l for l in txt.splitlines() if not l.startswith("#")]
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="\t", dtype=str).iloc[1:]      # drop the format row
        for _, r in df.iterrows():
            rows.append(dict(site_no=r["site_no"].zfill(8), huc8=str(r["huc_cd"]).zfill(8)[:8],
                             fips=str(r["state_cd"]).zfill(2) + str(r["county_cd"]).zfill(3), state=r["state_cd"].zfill(2)))
        print(f"  NWIS {min(i + 100, len(todo))}/{len(todo)}")
    adm = pd.concat([have, pd.DataFrame(rows)], ignore_index=True).drop_duplicates("site_no")
    adm.to_csv(f, index=False)
    return adm


# ---------------------------------------------------------------------------------------
# 2. USDM weekly statistics per area
# ---------------------------------------------------------------------------------------
def usdm_area(kind, aoi, start, end):
    """kind = 'HUC' or 'County'; cumulative percent area in D0..D4 per map week."""
    ep = {"HUC": "HUCStatistics", "County": "CountyStatistics"}[kind]
    p = {"aoi": aoi, "startdate": start.strftime("%m/%d/%Y"), "enddate": end.strftime("%m/%d/%Y"), "statisticsType": 1}
    if kind == "HUC":
        p["hucLevel"] = 8
    txt = get(f"{USDM}/{ep}/GetDroughtSeverityStatisticsByAreaPercent?{urlencode(p)}")
    try:
        js = json.loads(txt)
    except json.JSONDecodeError:
        raise RuntimeError(f"non-JSON response: {txt[:120]!r}")
    if not js:
        return pd.DataFrame()
    df = pd.DataFrame(js)
    df.columns = [c.lower() for c in df.columns]
    if "mapdate" not in df.columns:
        raise RuntimeError(f"unexpected fields: {list(df.columns)}")
    df["map_date"] = pd.to_datetime(df["mapdate"].astype(str).str[:10], format="mixed")
    return pd.DataFrame({"map_date": df["map_date"], **{D: df[D.lower()].astype(float) for D in ["D0", "D1", "D2", "D3", "D4"]}})


def fetch_all(kind, codes, start, end, fname, key):
    f = OUT / fname
    have = pd.read_parquet(f) if f.exists() else pd.DataFrame(columns=[key, "map_date", "D0", "D1", "D2", "D3", "D4"])
    done = set(have[key].astype(str))
    parts = [have]
    todo = [c for c in codes if c not in done]
    for k, code in enumerate(todo, 1):
        try:
            d = usdm_area(kind, code, start, end)
        except Exception as e:                     # noqa: BLE001
            print(f"  {kind} {code}: failed ({e})"); continue
        if d.empty:
            print(f"  {kind} {code}: no data"); continue
        d.insert(0, key, code); parts.append(d)
        if k % 20 == 0 or k == len(todo):
            print(f"  USDM {kind} {k}/{len(todo)}")
            pd.concat(parts, ignore_index=True).to_parquet(f, index=False)      # checkpoint
        time.sleep(0.3)
    allw = pd.concat(parts, ignore_index=True)
    allw.to_parquet(f, index=False)
    return allw


def from_csv(path, key, fname):
    """Convert a USDM 'Comprehensive Statistics' CSV export (traditional, percent area) to the parquet layout."""
    df = pd.read_csv(path, dtype=str)
    col = next(c for c in df.columns if c.upper() in ("HUC", "HUCID", "FIPS", "COUNTYFIPS"))
    out = pd.DataFrame({key: df[col].str.zfill(8 if key == "huc8" else 5),
                        "map_date": pd.to_datetime(df["MapDate"].astype(str), format="%Y%m%d")})
    for c in ["D0", "D1", "D2", "D3", "D4"]:
        out[c] = df[c].astype(float)
    out.to_parquet(OUT / fname, index=False)
    print(f"converted {path} -> {OUT / fname}: {out[key].nunique()} areas, {out.map_date.nunique()} weeks")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default=str(REPO / "data/basin_list.txt"))
    ap.add_argument("--start", default="2000-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--usdm_csv", nargs=2, metavar=("KIND", "PATH"), help="offline: HUC8|County  path/to/export.csv")
    a, _ = ap.parse_known_args()          # ignore Jupyter's own --f= argument
    OUT.mkdir(parents=True, exist_ok=True)
    sites = [l.strip().zfill(8) for l in open(a.sites) if l.strip()]
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)

    if a.usdm_csv:
        kind, path = a.usdm_csv
        from_csv(path, "huc8" if kind.upper() == "HUC8" else "fips",
                 "usdm_huc8_weekly.parquet" if kind.upper() == "HUC8" else "usdm_county_weekly.parquet")
        raise SystemExit

    print(f"{len(sites)} sites")
    adm = site_admin(sites)
    print(f"site admin: {len(adm)} sites, {adm.huc8.nunique()} HUC8s, {adm.fips.nunique()} counties")
    h = fetch_all("HUC", sorted(adm.huc8.unique()), start, end, "usdm_huc8_weekly.parquet", "huc8")
    print(f"HUC8 weekly: {h.huc8.nunique()} areas, {h.map_date.min().date()} .. {h.map_date.max().date()}")
    c = fetch_all("County", sorted(adm.fips.unique()), start, end, "usdm_county_weekly.parquet", "fips")
    print(f"county weekly: {c.fips.nunique()} areas")