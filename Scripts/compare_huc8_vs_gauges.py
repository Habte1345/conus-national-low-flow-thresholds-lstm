"""
Step 4: HUC8 thresholds (predicted from GEE Daymet forcing + GEE statics) vs the gauge-based LSTM thresholds of the
430 reference gauges, for every HUC8 that contains a gauge. Same models, so this isolates the effect of
(i) HUC8 areal forcing/statics instead of the CAMELS basin values and (ii) the 7 attributes filled with the training mean.

Outputs (data/huc_thresholds/):
  compare_ft_sites.csv      per gauge x percentile: gauge mu/sigma, HUC8 mu/sigma, log difference
  compare_ft_summary.csv    per percentile: bias, MAE, RMSE (log), R2, share within gauge sigma, share within HUC8 sigma
  compare_vt_summary.csv    same for VT over gauge x doy x percentile
  figures/compare_ft_scatter.png, figures/compare_vt_examples.png
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO      = Path("/bighome/hdagne1/conus-national-low-flow-thresholds-lstm")
HUC_DIR   = REPO / "data/huc_thresholds"
GAUGES    = REPO / "data/gee/gee_gauges.csv"                      # === CHECK: site_id + huc8 (or huc8_num) ===
SUM_FT    = REPO / "runs/run_2109_1723_ft_seed200/ft_lstm_seed200_summary.csv"
SUM_VT    = REPO / "runs/run_2209_0932_vt_seed200/vt_lstm_seed200_summary.csv"
FIG_DIR   = REPO / "figures"; FIG_DIR.mkdir(exist_ok=True)
NAMES     = ["P5", "P10", "P15", "P20", "P25", "P30"]

def stats(g, a, b, sa, sb):
    d = g[a] - g[b]
    return pd.Series({"n": len(g), "bias": d.mean(), "mae": d.abs().mean(), "rmse": np.sqrt((d ** 2).mean()),
                      "r2": np.corrcoef(g[a], g[b])[0, 1] ** 2, "within_gauge_sigma": (d.abs() < g[sa]).mean(),
                      "within_huc_sigma": (d.abs() < g[sb]).mean()})

# ---- gauge -> HUC8 ------------------------------------------------------------------------------------------
gg = pd.read_csv(GAUGES, dtype=str)
site_col = "site_id" if "site_id" in gg.columns else "site_no"
huc_col = "huc8" if "huc8" in gg.columns else "huc8_num"
gg["site"] = gg[site_col].str.zfill(8); gg["huc8"] = gg[huc_col].astype(str).str.zfill(8)
site2huc = gg.set_index("site")["huc8"]

# ---- FT ----------------------------------------------------------------------------------------------------
huc_ft = pd.read_csv(HUC_DIR / "huc_thresholds_ft_huc8.csv", dtype={"huc": str}).set_index("huc")
sft = pd.read_csv(SUM_FT, dtype={"site_no": str}); sft["site_no"] = sft.site_no.str.zfill(8)
sft["huc8"] = sft.site_no.map(site2huc)
sft = sft[sft.huc8.isin(huc_ft.index)]
rows = []
for _, r in sft.iterrows():
    h = huc_ft.loc[r.huc8]
    rows.append({"site_no": r.site_no, "huc8": r.huc8, "name": r["name"], "gauge_obs": r.obs_log, "gauge_mu": r.mu_log, "gauge_sigma": r.sigma_log,
                 "huc_mu": h[f"ft_{r['name']}_mu"], "huc_sigma": h[f"ft_{r['name']}_sigma"]})
cft = pd.DataFrame(rows); cft["diff"] = cft.huc_mu - cft.gauge_mu
cft.to_csv(HUC_DIR / "compare_ft_sites.csv", index=False)
summ = cft.groupby("name").apply(lambda g: stats(g, "huc_mu", "gauge_mu", "gauge_sigma", "huc_sigma")).reindex(NAMES)
summ.to_csv(HUC_DIR / "compare_ft_summary.csv")
print(f"FT: {cft.site_no.nunique()} gauges in {cft.huc8.nunique()} HUC8s\n", summ.round(3).to_string())
# vs the observed gauge thresholds as well (how much of the gauge-model error carries over)
obs = cft.groupby("name").apply(lambda g: stats(g, "huc_mu", "gauge_obs", "gauge_sigma", "huc_sigma")).reindex(NAMES)
print("\nHUC8 mu vs OBSERVED gauge thresholds:\n", obs[["bias", "mae", "rmse", "r2"]].round(3).to_string())

fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
for ax, n in zip(axes.ravel(), NAMES):
    g = cft[cft["name"] == n]
    ax.scatter(np.exp(g.gauge_mu), np.exp(g.huc_mu), s=10, alpha=.6)
    lim = [1e-3, 3]; ax.plot(lim, lim, "k--", lw=1); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    s = summ.loc[n]; ax.set_title(f"{n}: R²={s.r2:.2f}, RMSE={s.rmse:.2f} log, bias={s.bias:+.2f}")
    ax.set_xlabel("gauge-based LSTM-FT (mm/d)"); ax.set_ylabel("HUC8 LSTM-FT (mm/d)")
fig.suptitle("HUC8 thresholds from GEE forcing/statics vs gauge-based thresholds (HUC8s containing a reference gauge)")
fig.tight_layout(); fig.savefig(FIG_DIR / "compare_ft_scatter.png", dpi=150); plt.close(fig)

# ---- VT ----------------------------------------------------------------------------------------------------
huc_vt = pd.read_csv(HUC_DIR / "huc_thresholds_vt_huc8.csv", dtype={"huc": str})
svt = pd.read_csv(SUM_VT, dtype={"site_no": str}, usecols=["site_no", "name", "doy", "obs_log", "mu_log", "sigma_log"])
svt["site_no"] = svt.site_no.str.zfill(8); svt["huc8"] = svt.site_no.map(site2huc)
svt = svt[svt.huc8.isin(huc_vt.huc.unique())]
long = huc_vt.melt(id_vars=["huc", "doy"], value_vars=[f"vt_{n}_mu" for n in NAMES], var_name="v", value_name="huc_mu")
long["name"] = long.v.str.split("_").str[1]
sig = huc_vt.melt(id_vars=["huc", "doy"], value_vars=[f"vt_{n}_sigma" for n in NAMES], var_name="v", value_name="huc_sigma")
long["huc_sigma"] = sig.huc_sigma.to_numpy()
cvt = svt.merge(long[["huc", "doy", "name", "huc_mu", "huc_sigma"]], left_on=["huc8", "doy", "name"], right_on=["huc", "doy", "name"])
cvt["diff"] = cvt.huc_mu - cvt.mu_log
summ_vt = cvt.groupby("name").apply(lambda g: stats(g, "huc_mu", "mu_log", "sigma_log", "huc_sigma")).reindex(NAMES)
summ_vt.to_csv(HUC_DIR / "compare_vt_summary.csv")
print(f"\nVT: {cvt.site_no.nunique()} gauges x 366 doy\n", summ_vt.round(3).to_string())

ex = cvt[cvt["name"] == "P10"].groupby("site_no")["diff"].apply(lambda d: d.abs().mean()).sort_values()
picks = list(ex.index[[0, len(ex) // 3, 2 * len(ex) // 3, len(ex) - 1]])          # best .. worst agreement
fig, axes = plt.subplots(2, 2, figsize=(12, 7))
for ax, s in zip(axes.ravel(), picks):
    g = cvt[(cvt.site_no == s) & (cvt["name"] == "P10")].sort_values("doy")
    ax.plot(g.doy, np.exp(g.mu_log), label="gauge-based LSTM-VT"); ax.plot(g.doy, np.exp(g.huc_mu), label="HUC8 LSTM-VT")
    ax.plot(g.doy, np.exp(g.obs_log), "k:", lw=1, label="observed")
    ax.set_yscale("log"); ax.set_title(f"{s} (HUC8 {g.huc8.iloc[0]}), P10"); ax.set_xlabel("day of year"); ax.set_ylabel("mm/d")
axes[0, 0].legend()
fig.tight_layout(); fig.savefig(FIG_DIR / "compare_vt_examples.png", dpi=150); plt.close(fig)
print(f"figures -> {FIG_DIR}/compare_ft_scatter.png, compare_vt_examples.png")