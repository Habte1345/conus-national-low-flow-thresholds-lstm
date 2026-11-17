"""
GEE tables for the national HUC8 threshold product. Inputs: the two CSVs written by predict_huc8.py. Outputs (data/gee/):
  gee_huc8_thresholds_ft.csv   one row per HUC8: huc8, huc8_num, ft_<P>_mu/lo/hi (mm/d), ft_<P>_sigma (log)
  gee_huc8_thresholds_vt.csv   one row per HUC8 x day-of-year: huc8_num, doy, vt_<P>_mu/lo/hi (mm/d)
Upload both in the Code Editor: Assets -> New -> CSV file (plain tables, no geometry columns), into the app's asset folder.
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