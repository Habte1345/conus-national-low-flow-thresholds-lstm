"""Build a tiny CAMELS-format dataset + threshold table to smoke-test the pipeline."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
basins = ["01013500", "01022500", "02011400", "03010655"]
rng = np.random.default_rng(0)

# forcing + streamflow files
dates = pd.date_range("1980-01-01", "2014-12-31")
for b in basins:
    huc = b[:2]
    fdir = root / "basin_mean_forcing/nldas_extended" / huc
    fdir.mkdir(parents=True, exist_ok=True)
    doy = dates.dayofyear.values
    prcp = rng.gamma(0.7, 4.0, len(dates))
    srad = 250 + 120 * np.sin(2 * np.pi * (doy - 80) / 365) + rng.normal(0, 20, len(dates))
    tmax = 15 + 12 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 3, len(dates))
    tmin = tmax - 8
    vp = 800 + 500 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 50, len(dates))
    df = pd.DataFrame({"Year": dates.year, "Mnth": dates.month, "Day": dates.day, "Hr": 12,
                       "Dayl(s)": 43200, "PRCP(mm/day)": prcp, "SRAD(W/m2)": srad,
                       "SWE(mm)": 0.0, "Tmax(C)": tmax, "Tmin(C)": tmin, "Vp(Pa)": vp})
    with open(fdir / f"{b}_lump_nldas_forcing_leap.txt", "w") as f:
        f.write(" 45.0\n 300.0\n 2500000000\n")
        df.to_csv(f, sep=" ", index=False)

# attributes (2 files, ';' separated, gauge_id + huc_02 like CAMELS)
adir = root / "camels_attributes_v2.0"
adir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"gauge_id": basins, "huc_02": [int(b[:2]) for b in basins],
              "gauge_lat": rng.uniform(30, 48, 4), "gauge_lon": rng.uniform(-120, -70, 4),
              "elev_mean": rng.uniform(100, 2000, 4), "slope_mean": rng.uniform(10, 200, 4),
              "area_gages2": rng.uniform(100, 5000, 4)}).to_csv(adir / "camels_topo.txt", sep=";", index=False)
pd.DataFrame({"gauge_id": basins, "p_mean": rng.uniform(1, 5, 4), "pet_mean": rng.uniform(1, 5, 4),
              "aridity": rng.uniform(0.3, 3, 4), "frac_snow": rng.uniform(0, .5, 4),
              "q_mean": rng.uniform(0.1, 3, 4)}).to_csv(adir / "camels_clim.txt", sep=";", index=False)

# threshold table: ft, fq constants; vt per DOY
rows = []
for b in basins:
    base = rng.uniform(0.05, 0.6)
    for name, k in zip(["P2", "P5", "P10", "P20", "P30"], [0.6, 0.8, 1.0, 1.3, 1.6]):
        rows.append(dict(site_no=b, family="ft", name=name, doy=np.nan, value=base * k))
        for d in range(1, 366):
            rows.append(dict(site_no=b, family="vt", name=name, doy=d,
                             value=base * k * (1 + 0.5 * np.sin(2 * np.pi * (d - 60) / 365))))
    for name, k in zip(["Q1_2", "Q1_10", "Q7_2", "Q7_5", "Q7_10", "Q30_2", "Q30_5", "Q30_10"],
                       [0.5, 0.3, 0.55, 0.4, 0.33, 0.7, 0.5, 0.4]):
        rows.append(dict(site_no=b, family="fq", name=name, doy=np.nan, value=base * k))
# one deliberately missing FQ value to exercise NaN masking
th = pd.DataFrame(rows)
th = th[~((th.site_no == basins[0]) & (th.family == "fq") & (th.name == "Q30_10"))]
th.to_parquet(root / "thresholds.parquet", index=False)

(root.parent / "data").mkdir(exist_ok=True)
(root.parent / "data/basin_list.txt").write_text("\n".join(basins) + "\n")
print("synthetic data at", root)
