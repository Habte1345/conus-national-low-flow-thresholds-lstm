"""
Does driving the NLDAS-trained LSTMs with (converted) Daymet forcing reproduce the NLDAS results?
Uses the 430 training gauges, where CAMELS has BOTH products on disk. Two steps:

1. Fit the Daymet -> NLDAS correction on all basin-days 1980-2014: a per-variable, per-month linear map
   NLDAS = a + b * Daymet.  Written to data/daymet2nldas.json (used by predict_huc8.py).
2. Run LSTM-FT on every gauge with (a) NLDAS, (b) raw Daymet, (c) Daymet corrected, over WINDOW (the period both
   products cover in the local CAMELS copy; the Daymet files there end in Feb 1992),
   and compare the site-mean mu per percentile: bias, MAE, RMSE in log space, R2, and the share of sites where
   |mu_daymet - mu_nldas| < sigma_nldas (i.e. within the model's own predictive spread).

Outputs: data/daymet2nldas.json, data/check_daymet_vs_nldas_sites.csv, and a printed summary table.
Run on a GPU node:  python Scripts/check_daymet_vs_nldas.py
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import predict_huc8 as P
from data.datautils import SCALER, load_attributes

P.STRIDE = 7                                   # windows every 7 days: same site means, 7x faster
CAMELS_ROOT = P.REPO / "data/CAMELS_US"        # === CHECK: cfg camels_root ===
BASIN_LIST  = P.REPO / "data/basin_list.txt"
OUT_SITES   = P.REPO / "data/check_daymet_vs_nldas_sites.csv"
FAMILIES    = ["ft"]                           # add "vt" for the doy product (slower)
WINDOW      = (pd.Timestamp("1981-01-01"), pd.Timestamp("1992-02-28"))   # window END dates for the model test: the period both products cover
NEED        = (WINDOW[0] - pd.DateOffset(days=364), WINDOW[1])            # incl. the 365-day warm-up; a gauge must cover it in both products

NAMEMAP = {"prcp": "PRCP", "srad": "SRAD", "tmax": "Tmax", "tmin": "Tmin", "vp": "Vp", "dayl": "Dayl"}

def load_camels(basin, product):
    sub = "daymet" if product == "daymet" else "nldas_extended"
    files = list((CAMELS_ROOT / "basin_mean_forcing" / sub).glob(f"**/{basin}_*forcing_leap.txt"))
    if not files:
        raise FileNotFoundError(f"{product} forcing for {basin} not under {CAMELS_ROOT}")
    with open(files[0]) as f:
        hdr = next((i for i, line in enumerate(f) if line.lstrip().lower().startswith("year")), 3)   # CAMELS: header on line 4
    df = pd.read_csv(files[0], sep=r"\s+", header=hdr)
    if "Year" not in df.columns:
        raise ValueError(f"unexpected layout in {files[0]}: columns {list(df.columns)[:6]}")
    df.index = pd.to_datetime(dict(year=df.Year, month=df.Mnth, day=df.Day), errors="coerce")   # CAMELS quirk: some rows have day 00
    df = df[df.index.notna() & ~df.index.duplicated()].sort_index()
    df = df.rename(columns={c: NAMEMAP.get(c.split("(")[0].lower(), c) for c in df.columns})
    df = df[P.VARS].apply(pd.to_numeric, errors="coerce").astype(np.float64)                   # '-' placeholders -> NaN
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="D")).ffill().bfill()         # continuous daily calendar, gaps filled
    return df.loc[P.PERIOD[0]:P.PERIOD[1]]


def main():
    basins = [b.strip() for b in open(BASIN_LIST) if b.strip()]
    print(f"{len(basins)} gauges")

    # ---- 1. correction -----------------------------------------------------------------------------------------
    pairs = {v: [] for v in P.DYN}                              # per variable: rows (month, daymet, nldas)
    cache, skipped = {}, []
    for b in basins:
        try:
            d, n = load_camels(b, "daymet"), load_camels(b, "nldas")
        except Exception as e:
            skipped.append(b); print(f"skip {b}: {e}"); continue
        if not (d.index.min() <= NEED[0] and d.index.max() >= NEED[1] and n.index.min() <= NEED[0] and n.index.max() >= NEED[1]):
            skipped.append(b); print(f"skip {b}: daymet {d.index.min().date()}..{d.index.max().date()}, nldas {n.index.min().date()}..{n.index.max().date()} (truncated file)"); continue
        idx = d.index.intersection(n.index)
        d, n = d.loc[idx], n.loc[idx]
        cache[b] = (d, n)
        m = idx.month.to_numpy()
        for v in P.DYN:
            pairs[v].append(np.c_[m, d[v].to_numpy(), n[v].to_numpy()])
    corr, diag = {}, []
    for j, v in enumerate(P.DYN):
        xy = np.concatenate(pairs[v]); xy = xy[np.isfinite(xy).all(1)]
        m, x, y = xy[:, 0].astype(int), xy[:, 1], xy[:, 2]
        a, bb = np.zeros(12), np.ones(12)
        for k in range(12):
            sel = m == k + 1
            bb[k], a[k] = np.polyfit(x[sel], y[sel], 1)
        corr[v] = {"a": a.tolist(), "b": bb.tolist()}
        fitted = a[m - 1] + bb[m - 1] * x
        sd = SCALER["input_stds"][j]
        diag.append({"var": v, "bias_raw": (x - y).mean(), "rmse_raw": np.sqrt(((x - y) ** 2).mean()),
                     "rmse_raw_in_scaler_std": np.sqrt(((x - y) ** 2).mean()) / sd,
                     "rmse_corrected": np.sqrt(((fitted - y) ** 2).mean()),
                     "rmse_corrected_in_scaler_std": np.sqrt(((fitted - y) ** 2).mean()) / sd,
                     "r2": np.corrcoef(x, y)[0, 1] ** 2})
    basins = [b for b in basins if b not in skipped]
    if skipped:
        print(f"{len(skipped)} basins skipped (unreadable forcing files): {skipped}")
    json.dump(corr, open(P.CORR_FILE, "w"), indent=1)
    print("\nraw Daymet vs NLDAS, all gauge-days 1980-2014 (rmse_corrected = after the monthly linear map):")
    print(pd.DataFrame(diag).round(3).to_string(index=False))
    print(f"correction written -> {P.CORR_FILE}")

    # ---- 2. model agreement ------------------------------------------------------------------------------------
    rows = []
    for fam in FAMILIES:
        run = P.load_run(fam)
        attr = load_attributes(str(P.RUNS[fam] / "attributes.db"), basins, drop_lat_lon=True)
        for i, b in enumerate(basins):
            d, n = cache[b]
            xs = P.static_vector(run, attr.loc[b])
            res = {}
            for tag, daily in [("nldas", n), ("daymet_raw", d), ("daymet_corr", P.daymet_to_nldas(d, corr))]:
                mu, sg, dates = P.predict_windows(run, daily, xs, window=WINDOW)
                res[tag] = (mu.mean(0), sg.mean(0))
            for k, name in enumerate(run["names"]):
                rows.append({"family": fam, "site_no": b, "name": name,
                             "mu_nldas": res["nldas"][0][k], "sigma_nldas": res["nldas"][1][k],
                             "mu_daymet_raw": res["daymet_raw"][0][k], "mu_daymet_corr": res["daymet_corr"][0][k]})
            if i % 50 == 0:
                print(f"  {fam}: {i}/{len(basins)}")
    df = pd.DataFrame(rows); df.to_csv(OUT_SITES, index=False)

    print("\nSite-mean mu (log mm/day): Daymet-driven vs NLDAS-driven LSTM")
    for tag in ["mu_daymet_raw", "mu_daymet_corr"]:
        e = df[tag] - df["mu_nldas"]
        s = df.groupby(["family", "name"]).apply(lambda g: pd.Series({
            "bias": (g[tag] - g["mu_nldas"]).mean(), "mae": (g[tag] - g["mu_nldas"]).abs().mean(),
            "rmse": np.sqrt(((g[tag] - g["mu_nldas"]) ** 2).mean()),
            "r2": np.corrcoef(g[tag], g["mu_nldas"])[0, 1] ** 2,
            "within_1sigma": ((g[tag] - g["mu_nldas"]).abs() < g["sigma_nldas"]).mean()}))
        print(f"\n[{tag}]  overall MAE {e.abs().mean():.3f}, within 1 sigma {((e.abs() < df.sigma_nldas).mean()):.2%}")
        print(s.round(3).to_string())
    print(f"\nper-site table -> {OUT_SITES}")

if __name__ == "__main__":
    main()