"""
GEE tables for the national HUC8 threshold product. Inputs: the two CSVs written by predict_huc8.py. Outputs (data/gee/):
  gee_huc8_thresholds_ft.csv   one row per HUC8: huc8, huc8_num, ft_<P>_mu/lo/hi (mm/d), ft_<P>_sigma (log)
  gee_huc8_thresholds_vt.csv   one row per HUC8 x day-of-year: huc8_num, doy, vt_<P>_mu/lo/hi (mm/d)   (click charts)
  gee_huc8_thresholds_vt_month.csv  one row per HUC8: huc8_num, vt_<P>_m01..m12 mu/lo/hi = monthly means (mm/d)  (maps; fast)
Upload all three in the Code Editor: Assets -> New -> CSV file (plain tables, no geometry columns), into the app's asset folder.
"""
from pathlib import Path
import numpy as np, pandas as pd

HUC_DIR = Path("/bighome/hdagne1/conus-national-low-flow-thresholds-lstm/data/huc_thresholds")
OUT     = Path("/bighome/hdagne1/conus-national-low-flow-thresholds-lstm/data/gee"); OUT.mkdir(parents=True, exist_ok=True)
NAMES   = ["P5", "P10", "P15", "P20", "P25", "P30"]

ft = pd.read_csv(HUC_DIR / "huc_thresholds_ft_huc8.csv", dtype={"huc": str})
F = pd.DataFrame({"huc8": ft.huc, "huc8_num": ft.huc.astype(int)})
for n in NAMES:
    F[f"ft_{n}_mu"], F[f"ft_{n}_lo"], F[f"ft_{n}_hi"] = ft[f"ft_{n}_mmd"], ft[f"ft_{n}_lo"], ft[f"ft_{n}_hi"]
    F[f"ft_{n}_sigma"] = ft[f"ft_{n}_sigma"]
F.round(5).to_csv(OUT / "gee_huc8_thresholds_ft.csv", index=False)
print(f"gee_huc8_thresholds_ft.csv: {len(F)} HUC8s, {F.shape[1]} columns")

vt = pd.read_csv(HUC_DIR / "huc_thresholds_vt_huc8.csv", dtype={"huc": str})
V = pd.DataFrame({"huc8_num": vt.huc.astype(int), "doy": vt.doy.astype(int)})
for n in NAMES:
    V[f"vt_{n}_mu"], V[f"vt_{n}_lo"], V[f"vt_{n}_hi"] = vt[f"vt_{n}_mmd"], vt[f"vt_{n}_lo"], vt[f"vt_{n}_hi"]
V.round(5).to_csv(OUT / "gee_huc8_thresholds_vt.csv", index=False)
print(f"gee_huc8_thresholds_vt.csv: {len(V)} rows ({V.huc8_num.nunique()} HUC8s x {V.doy.nunique()} doy)")

# monthly means per HUC8 (12 x 6 x 3 = 216 value columns) -> the VT map layers need no per-day filtering
doy = vt.doy.to_numpy()
month = np.searchsorted(np.array([31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 366]), doy, side="left") + 1   # 1..12 (365-day calendar; doy 366 -> Dec)
vt["month"] = np.clip(month, 1, 12)
M = pd.DataFrame({"huc8_num": sorted(vt.huc.astype(int).unique())}).set_index("huc8_num")
g = vt.assign(huc8_num=vt.huc.astype(int)).groupby(["huc8_num", "month"])
for n in NAMES:
    for k, src in [("mu", f"vt_{n}_mmd"), ("lo", f"vt_{n}_lo"), ("hi", f"vt_{n}_hi")]:
        piv = g[src].mean().unstack("month")
        for m in range(1, 13):
            M[f"vt_{n}_m{m:02d}_{k}"] = piv[m]
M.round(5).reset_index().to_csv(OUT / "gee_huc8_thresholds_vt_month.csv", index=False)
print(f"gee_huc8_thresholds_vt_month.csv: {len(M)} HUC8s, {M.shape[1]} columns")