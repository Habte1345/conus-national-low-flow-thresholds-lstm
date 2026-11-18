"""Small mock of the NGIAB layout used by the pipeline (for smoke tests only).

<root>/Data/CAMELS_US_ngiab_full/gage-<id>/config/<id>_subset.gpkg   divides, divide-attributes, flowpaths, nexus
<root>/Data/CAMELS_US_ngiab_full/gage-<id>/forcings/forcings.nc      ids, Time, hourly AORC variables
<root>/Data/usgs_streamflow_ref/<id>.parquet                         flow_m3s (daily)
<root>/NeuralNGEN/interior_gauges.csv, obs_cache_interior/<id>.csv    interior gauges
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

VARS = ["APCP_surface", "DLWRF_surface", "DSWRF_surface", "PRES_surface", "SPFH_2maboveground",
        "TMP_2maboveground", "UGRD_10maboveground", "VGRD_10maboveground"]
ATTR_COLS = (["bexp_soil_layers_stag_%d" % i for i in range(1, 5)] + ["dksat_soil_layers_stag_%d" % i for i in range(1, 5)]
             + ["psisat_soil_layers_stag_%d" % i for i in range(1, 5)] + ["cwpvt", "mfsno", "mp", "refkdt", "slope_1km"]
             + ["smcmax_soil_layers_stag_%d" % i for i in range(1, 5)] + ["smcwlt_soil_layers_stag_%d" % i for i in range(1, 5)]
             + ["vcmx25", "Coeff", "Zmax", "Expon", "impervious", "elevation", "slope", "aspect"])


def make(root, n_basins=6, n_fp=12, start="2000-01-01", end="2004-12-31", seed=0):
    rng = np.random.default_rng(seed)
    root = Path(root)
    hours = pd.date_range(start, f"{end} 23:00", freq="h")
    days = pd.date_range(start, end, freq="D")
    interior = []
    for b in range(n_basins):
        gid = f"0{b + 1}000000"
        gd = root / "Data/CAMELS_US_ngiab_full" / f"gage-{gid}"
        (gd / "config").mkdir(parents=True, exist_ok=True); (gd / "forcings").mkdir(parents=True, exist_ok=True)
        base = 1000 * (b + 1)
        ids = np.arange(base + 1, base + n_fp + 1)
        down = {ids[i]: (ids[(i - 1) // 2] if i > 0 else None) for i in range(n_fp)}   # binary tree, root ids[0]
        area = rng.uniform(3, 15, n_fp)
        wet = 0.5 + b / n_basins                                                       # basin wetness
        con = sqlite3.connect(gd / "config" / f"{gid}_subset.gpkg")
        pd.DataFrame({"divide_id": [f"cat-{i}" for i in ids], "areasqkm": area, "id": [f"wb-{i}" for i in ids]}) \
            .to_sql("divides", con, index=False)
        att = pd.DataFrame(rng.uniform(0.1, 1.0, (n_fp, len(ATTR_COLS))), columns=ATTR_COLS)
        att["aspect"] = rng.uniform(0, 360, n_fp); att["Coeff"] = wet
        att.insert(0, "divide_id", [f"cat-{i}" for i in ids])
        att.to_sql("divide-attributes", con, index=False)
        pd.DataFrame({"id": [f"wb-{i}" for i in ids], "toid": [f"nex-{i}" for i in ids]}).to_sql("flowpaths", con, index=False)
        pd.DataFrame({"id": [f"nex-{i}" for i in ids],
                      "toid": [f"wb-{down[i]}" if down[i] is not None else None for i in ids]}).to_sql("nexus", con, index=False)
        con.close()

        # hourly forcing
        nt = len(hours); doy = hours.dayofyear.values
        storm = (rng.random((n_fp, nt)) < 0.01) * rng.gamma(1.0, 2e-4 * wet, (n_fp, nt))
        prcp = np.apply_along_axis(lambda x: np.convolve(x, np.ones(6) / 6, "same"), 1, storm)
        temp = 283 + 12 * np.sin(2 * np.pi * (doy - 110) / 365)[None, :] + rng.normal(0, 2, (n_fp, nt))
        data = {"APCP_surface": prcp, "TMP_2maboveground": temp}
        for v in VARS:
            if v not in data:
                data[v] = rng.normal(1, 0.1, (n_fp, nt))
        tsec = ((hours - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).values.astype(np.int64)
        ds = xr.Dataset({v: (("catchment-id", "time"), data[v].astype(np.float32)) for v in VARS})
        ds["ids"] = (("catchment-id",), np.array([f"cat-{i}" for i in ids]))
        ds["Time"] = (("catchment-id", "time"), np.repeat(tsec[None, :], n_fp, 0))
        ds["APCP_surface"].attrs["units"] = "kg/m^2/s"
        ds.to_netcdf(gd / "forcings" / "forcings.nc")

        # daily runoff per divide (linear reservoir), flow at a flowpath = sum over upstream divides
        p_d = pd.DataFrame(prcp.T, index=hours).resample("D").mean().values.T * 86400       # mm/d
        r = np.zeros_like(p_d); s = np.zeros(n_fp); k = 0.03 + 0.02 * wet
        for t in range(p_d.shape[1]):
            s = s + p_d[:, t]; r[:, t] = k * s; s = s - r[:, t]
        upstream = {i: [i] for i in ids}
        for i in ids[::-1]:
            if down[i] is not None:
                upstream[down[i]] += upstream[i]
        pos = {i: j for j, i in enumerate(ids)}

        def q_m3s(f):
            u = [pos[x] for x in set(upstream[f])]
            return (r[u] * area[u][:, None]).sum(0) / 86.4

        sfd = root / "Data/usgs_streamflow_ref"; sfd.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"flow_m3s": q_m3s(ids[0])}, index=pd.Index(days, name="datetime")).to_parquet(sfd / f"{gid}.parquet")
        od = root / "NeuralNGEN/obs_cache_interior"; od.mkdir(parents=True, exist_ok=True)
        for f in (ids[1], ids[2]):
            ig = f"{gid[:2]}{f:06d}"
            interior.append(dict(outlet_gage=gid, interior_gage=ig, flowpath=int(f)))
            pd.DataFrame({"datetime": days, "Q_m3s": q_m3s(f)}).to_csv(od / f"{ig}.csv", index=False)
        (root / "NeuralNGEN/basin_lists").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(interior).to_csv(root / "NeuralNGEN/interior_gauges.csv", index=False)
    (root / "NeuralNGEN/basin_lists/basins.txt").write_text(
        "\n".join(f"0{b + 1}000000" for b in range(n_basins)) + "\n")
    return root


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mock_ngiab")
