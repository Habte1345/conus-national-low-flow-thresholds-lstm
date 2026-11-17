"""
Predicted vs calculated thresholds for one LSTM run — axes in real threshold units.

Usage (script):    python src/compare_thresholds.py runs/run_2109_1000_ft_seed200
Usage (notebook):  set RUN (and optionally AREA_FILE) below and run the cell.

Units: mm/day by default (what the model is trained on). Set AREA_FILE to the GAGES-II parquet
(with DRAIN_SQKM) to show everything in m3/s instead — the unit the USGS thresholds were
calculated in. All axes are linear, in actual threshold values.

Reads  <run>/<family>_lstm_seed*_summary.csv
Writes figures/<family>_1_scatter_cdf.png    site-level agreement + national distribution
       figures/<family>_2_errors_pit.png     error spread + uncertainty calibration
       figures/<family>_3_rank.png           every site, sorted, with the 90% predictive band
       figures/<family>_metrics.csv
"""
import sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, FixedLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import norm, ks_2samp

RUN = Path(r"F:\HPC\conus-national-low-flow-thresholds-lstm\runs\run_2109_1000_ft_seed200")
AREA_FILE = None   # e.g. r"F:\...\conus-lowflow-hydro-drought-thresholds\data\gage-II\gages_ii.parquet"  -> m3/s

# ---------------------------------------------------------------------------------------
csv = glob.glob(str(RUN / "*_summary.csv"))[0]
s = pd.read_csv(csv, dtype={"site_no": str}).dropna(subset=["obs_log", "mu_log"])
family = Path(csv).name.split("_")[0]
if "doy" in s.columns:                       # VT: collapse DOY to the site level for these figures
    s = s.groupby(["site_no", "name"], as_index=False)[["obs_log", "mu_log", "sigma_log", "crps_log"]].mean()
ORDER = ["P5","P10","P15","P20","P25","P30","Q1_2","Q1_10","Q7_2","Q7_5","Q7_10","Q30_2","Q30_5","Q30_10"]
names = [n for n in ORDER if n in set(s.name)]
n_sites = s.site_no.nunique()

# real-unit values (log-normal predictive distribution: mu = median, band = exp(mu ± 1.645 sigma))
UNIT = "mm/d"
if AREA_FILE:
    g = pd.read_parquet(AREA_FILE); g["site_no"] = g["site_no"].astype(str).str.zfill(8)
    area = s["site_no"].map(g.set_index("site_no")["DRAIN_SQKM"].astype(float))
    conv = np.log(area / 86.4)                # mm/d -> m3/s in log space
    s["obs_log"] += conv; s["mu_log"] += conv
    UNIT = "m³/s"
s["obs"] = np.exp(s.obs_log); s["mu"] = np.exp(s.mu_log)
s["lo90"] = np.exp(s.mu_log - 1.645 * s.sigma_log); s["hi90"] = np.exp(s.mu_log + 1.645 * s.sigma_log)
s["ratio"] = s["mu"] / s["obs"]                              # LSTM / calculated
s["err10"] = np.log10(s["ratio"])
s["pit"] = norm.cdf((s.obs_log - s.mu_log) / s.sigma_log)

CMAP = plt.get_cmap("jet")
BLUE, RED = CMAP(0.0), CMAP(1.0)
REDBAND = (*RED[:3], 0.25)
INK = "#222222"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold",
                     "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "legend.fontsize": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})
out = Path("figures"); out.mkdir(exist_ok=True)
ncol = len(names)
W = 3.4 * ncol
BOX = dict(fc="white", ec="none", alpha=0.95, pad=2)
HSPACE, WSPACE = 0.12, 0.06   # gap between rows / columns (fraction of a panel); bigger = more space


def fmt(v, _):
    return f"{v:g}"


def logaxis(ax, which="both"):
    for a in ([ax.xaxis, ax.yaxis] if which == "both" else [ax.xaxis] if which == "x" else [ax.yaxis]):
        a.set_major_locator(LogLocator(base=100 if a is ax.xaxis else 10))
        a.set_major_formatter(FuncFormatter(fmt))
        a.set_minor_formatter(FuncFormatter(lambda v, p: ""))


def metrics(d):
    r = np.corrcoef(d.obs_log, d.mu_log)[0, 1]
    rmse = np.sqrt((d.err10 ** 2).mean())
    within2 = (d.err10.abs() <= np.log10(2)).mean() * 100
    cov90 = ((d.pit > 0.05) & (d.pit < 0.95)).mean() * 100
    ks = ks_2samp(d.obs_log, d.mu_log)
    return r, d.ratio.median(), rmse, within2, cov90, ks.statistic, ks.pvalue


# ======================= Fig 1a: scatter =================================================
fig, axes = plt.subplots(1, ncol, figsize=(W, 4.0), constrained_layout=True)
fig.get_layout_engine().set(wspace=WSPACE)
for j, n in enumerate(names):
    d = s[s.name == n]
    r, medr, rmse, w2, cov, ksd, ksp = metrics(d)
    ax = axes[j]
    lo, hi = 0.0, 1.05 * max(d.obs.max(), d.mu.max())
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1, ls="--")
    sc = ax.scatter(d.obs, d.mu, c=d.err10, cmap=CMAP, vmin=-1, vmax=1, s=14, alpha=0.85,
                    linewidths=0.2, edgecolors="k")
    ax.text(0.03, 0.97, f"r = {r:.3f}",
            transform=ax.transAxes, va="top", fontsize=14, color=INK,)
    ax.set_title(n); ax.set_xlabel(f"USGS  [{UNIT}]", fontsize=14)
    if j == 0: ax.set_ylabel(f"LSTM [{UNIT}]", fontsize=14)
cb = fig.colorbar(sc, ax=axes.tolist(), pad=0.01, shrink=0.9, aspect=25)
cb.set_ticks([-1, -0.5, 0, 0.5, 1]); cb.set_ticklabels(["0.1", "0.3", "1", "3", "10"])
cb.set_label("Fixed-thresholds [mm/d]", fontsize=16)
# fig.savefig(out / f"{family}_1a_scatter.png", dpi=200)

# ======================= Fig 1b: cross-site CDF ===========================================
fig, axes = plt.subplots(1, ncol, figsize=(W, 4.0), constrained_layout=True)
fig.get_layout_engine().set(wspace=WSPACE)
for j, n in enumerate(names):
    d = s[s.name == n]
    r, medr, rmse, w2, cov, ksd, ksp = metrics(d)
    ax = axes[j]
    for col, c, lab in [("obs", BLUE, "calculated (USGS)"), ("mu", RED, "LSTM")]:
        v = np.sort(d[col].values); ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, lw=2.2, label=lab)
    ax.text(0.03, 0.97, f"KS = {ksd:.3f}\np = {ksp:.2f}", transform=ax.transAxes, va="top", fontsize=14, color=INK, bbox=BOX)
    ax.set_title(n); ax.set_xlabel(f"threshold  [{UNIT}]")
    if j == 0: ax.set_ylabel("fraction of sites")
axes[0].legend(loc="center right", frameon=False, fontsize=14)
# fig.savefig(out / f"{family}_1b_cdf.png", dpi=200)

# ======================= Fig 2: ratio distribution ========================================
fig, axes = plt.subplots(1, ncol, figsize=(W, 4.0), constrained_layout=True)
fig.get_layout_engine().set(wspace=WSPACE)
edges = np.arange(0, 5.01, 0.25)          # ratio bins, linear
for j, n in enumerate(names):
    d = s[s.name == n]
    r, medr, rmse, w2, cov, ksd, ksp = metrics(d)
    ax = axes[j]
    cnt, _ = np.histogram(d.ratio.clip(0, 5), bins=edges)
    mids = 0.5 * (edges[1:] + edges[:-1])
    ax.bar(mids, cnt, width=0.25, color=CMAP(np.clip(mids / 2, 0, 1)), edgecolor="#555555", lw=0.5)
    ax.axvline(1, color=INK, lw=1, ls="--"); ax.axvline(medr, color=INK, lw=1.6)
    ax.axvspan(0.5, 2, color="#dddddd", alpha=0.35, lw=0)
    ax.set_title(n)
    # ax.text(0.03, 0.97, f"median ratio {medr:.2f}%", transform=ax.transAxes, va="top", fontsize=14, color=INK, bbox=BOX)
    # if j == 0: ax.set_ylabel("sites")
    # ax.set_xlabel("LSTM / calculated")
# fig.savefig(out / f"{family}_2_ratio.png", dpi=200)

# ======================= Fig 3: every site, sorted, with 90% predictive band ==============
fig, axes = plt.subplots(1, ncol, figsize=(W, 4.6), sharey=True, constrained_layout=True)
fig.get_layout_engine().set(wspace=WSPACE)
for j, n in enumerate(names):
    d = s[s.name == n].sort_values("obs").reset_index(drop=True)
    x = np.arange(len(d))
    ax = np.atleast_1d(axes)[j]
    ax.fill_between(x, d.lo90, d.hi90, color=REDBAND, lw=0)
    ax.plot(x, d.mu, color=RED, lw=1.1)
    ax.plot(x, d.obs, color=BLUE, lw=1.8)
    ax.set_title(n); ax.set_xlabel("sites, sorted by calculated value")
    if j == 0: ax.set_ylabel(f"threshold  [{UNIT}]")
handles = [Line2D([], [], color=BLUE, lw=1.8, label="calculated (USGS)"), Line2D([], [], color=RED, lw=1.1, label="LSTM median"),
           Patch(fc=REDBAND, label="LSTM 90% band")]
np.atleast_1d(axes)[0].legend(handles=handles, loc="upper left", frameon=False)
# fig.suptitle(f"LSTM-{family.upper()} — all {n_sites} sites with predictive uncertainty", fontweight="bold", fontsize=16)
# fig.savefig(out / f"{family}_3_rank.png", dpi=200)

# ======================= summary table =====================================================
rows = [(n, *metrics(s[s.name == n]), s[s.name == n].crps_log.mean()) for n in names]
tab = pd.DataFrame(rows, columns=["name", "r_log", "median_ratio", "RMSE_log10", "within_x2_%", "cov90_%", "KS", "KS_p", "CRPS_log"])
# print(tab.round(3).to_string(index=False))
tab.to_csv(out / f"{family}_metrics.csv", index=False)
# print(f"units: {UNIT} — saved figures to", out.resolve())
plt.show()