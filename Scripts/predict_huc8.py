"""
HUC8 low-flow thresholds from the trained LSTM-FT and LSTM-VT, driven by the GEE Daymet HUC8 forcing.

Mirrors src/main.py::evaluate exactly:
  * inputs  = 5 forcings (PRCP, SRAD, Tmax, Tmin, Vp) normalised with datautils.SCALER (+ sin/cos DOY for VT),
              concatenated with the 27 CAMELS attributes standardised by the TRAINING-basin mean/std (attributes.db)
  * samples = 365-day windows ending on every day of the validation period (cfg val_start..val_end)
  * outputs = (mu, sigma) un-standardised with cfg target_means/target_stds -> log(mm/day)
  * FT: mean of mu and sigma over all windows;  VT: mean per day-of-year of the window end date
Daymet -> NLDAS: per-variable / per-month linear correction from data/daymet2nldas.json (written by check_daymet_vs_nldas.py
from the CAMELS gauges, where both products exist). Without that file the raw Daymet values are used.

Inputs : data/huc_forcing/forcing_huc8_<year>.csv (GEE wide monthly), data/huc_statics/statics_huc8.csv
Outputs: data/huc_thresholds/huc_thresholds_ft_huc8.csv, huc_thresholds_vt_huc8.csv
"""
import glob, importlib.util, json, sys
from pathlib import Path
import h5py, numpy as np, pandas as pd, torch

REPO        = Path("/bighome/hdagne1/conus-national-low-flow-thresholds-lstm")
FORCING_DIR = REPO / "data/huc_forcing"
STATICS_CSV = REPO / "data/huc_statics/statics_huc8.csv"
RUNS        = {"ft": REPO / "runs/run_2109_1723_ft_seed200", "vt": REPO / "runs/run_2209_0932_vt_seed200"}
OUT_DIR     = REPO / "data/huc_thresholds"
CORR_FILE   = REPO / "data/daymet2nldas.json"          # optional, from check_daymet_vs_nldas.py
LEVEL       = "huc8"
PERIOD      = ("1980-01-01", "2014-12-31")             # forcing loaded (must cover val_start-364d .. val_end)
CLIM_PERIOD = ("1989-10-01", "2009-09-30")             # CAMELS climate-index window (Addor et al. 2017)
STRIDE      = 1                                        # 1 = every window as in evaluate(); 7 = 7x faster, same means
BATCH       = 512
CHECKPOINT_EVERY = 50                                  # units per CSV append; a killed run resumes where it stopped
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(REPO))
from data.datautils import SCALER, LOG_EPS, load_attributes, doy_features            # noqa: E402
_spec = importlib.util.spec_from_file_location("lstm_main", REPO / "src/main.py")
_main = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_main)
Model = _main.Model

DYN  = ["PRCP", "SRAD", "Tmax", "Tmin", "Vp"]                                         # SCALER order
VARS = DYN + ["Dayl"]

# ---- GEE wide monthly -> daily ----------------------------------------------------------------------------
def read_forcing_wide(path):
    w = pd.read_csv(path, dtype={"huc": str})
    ids = w[["huc", "lat", "year", "month"]]
    out = ids.loc[ids.index.repeat(31)].reset_index(drop=True)
    out["day"] = np.tile(np.arange(1, 32), len(w))
    for v in VARS:
        cols = [f"{v}_d{d:02d}" for d in range(1, 32)]
        out[v] = w.reindex(columns=cols).to_numpy(dtype=np.float32).ravel()
    out["date"] = pd.to_datetime(out[["year", "month", "day"]], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values(["huc", "date"]).reset_index(drop=True)
    out[VARS] = out.groupby("huc", sort=False)[VARS].ffill()                          # Daymet has no Dec 31 in leap years
    out[VARS] = out.groupby("huc", sort=False)[VARS].bfill()
    return out[["huc", "date"] + VARS]

def load_forcing():
    files = [f for f in sorted(glob.glob(str(FORCING_DIR / f"forcing_{LEVEL}_*.csv")))
             if PERIOD[0][:4] <= Path(f).stem[-4:] <= PERIOD[1][:4]]
    if not files:
        raise SystemExit(f"no forcing_{LEVEL}_<year>.csv in {FORCING_DIR}")
    forc = pd.concat([read_forcing_wide(f) for f in files], ignore_index=True)
    forc = forc[(forc.date >= PERIOD[0]) & (forc.date <= PERIOD[1])].sort_values(["huc", "date"]).reset_index(drop=True)
    print(f"forcing: {forc.huc.nunique()} {LEVEL} units, {forc.date.min().date()} .. {forc.date.max().date()}, {len(files)} files")
    return forc

# ---- Daymet -> NLDAS-equivalent forcing ---------------------------------------------------------------------
def load_correction():
    return json.load(open(CORR_FILE)) if CORR_FILE.exists() else None

def daymet_to_nldas(df, corr=None):
    """df: daily with VARS (Daymet). Per-variable, per-month linear map NLDAS = a + b*Daymet fitted on the CAMELS gauges
    (check_daymet_vs_nldas.py). No SRAD daylight->24h conversion: CAMELS nldas_extended SRAD is a daylight-period mean too."""
    df = df.copy()
    if corr:
        m = df["date"].dt.month.to_numpy() if "date" in df else df.index.month.to_numpy()
        for v in DYN:
            if v in corr:
                a = np.asarray(corr[v]["a"])[m - 1]; b = np.asarray(corr[v]["b"])[m - 1]
                df[v] = a + b * df[v].to_numpy()
    return df

# ---- CAMELS climate indices (Addor et al. 2017) from the daily forcing -----------------------------------------
def pet_priestley_taylor(df):
    t = 0.5 * (df.Tmax + df.Tmin)
    es = 0.6108 * np.exp(17.27 * t / (t + 237.3)); delta = 4098 * es / (t + 237.3) ** 2
    rn = df.SRAD * 0.0864 * 0.77                                                      # 24-h mean W/m2 -> MJ/m2/d, net ~ 0.77 * incoming
    return np.clip(1.26 * delta / (delta + 0.066) * rn / 2.45, 0, None)

def climate_indices(df):
    df = df[(df.date >= CLIM_PERIOD[0]) & (df.date <= CLIM_PERIOD[1])]
    p = df.PRCP.to_numpy(); t = 0.5 * (df.Tmax + df.Tmin).to_numpy(); pet = pet_priestley_taylor(df).to_numpy()
    p_mean, pet_mean = p.mean(), pet.mean()
    doy = df.date.dt.dayofyear.to_numpy()
    def sine_fit(x):
        w = 2 * np.pi * (doy - 1) / 365.25; A = np.c_[np.ones_like(w), np.sin(w), np.cos(w)]
        c = np.linalg.lstsq(A, x, rcond=None)[0]; return np.hypot(c[1], c[2]), np.arctan2(c[2], c[1])
    dp, sp = sine_fit(p); _, st = sine_fit(t)
    def runs(mask):
        r, n = [], 0
        for k in mask:
            if k: n += 1
            elif n: r.append(n); n = 0
        if n: r.append(n)
        return r
    hi = p >= 5 * p_mean; lo = p < 1; years = len(p) / 365.25
    return {"p_mean": p_mean, "pet_mean": pet_mean, "aridity": pet_mean / p_mean,
            "p_seasonality": (dp / p_mean) * np.sign(np.cos(sp - st)), "frac_snow": p[t < 0].sum() / p.sum(),
            "high_prec_freq": hi.sum() / years, "high_prec_dur": np.mean(runs(hi)) if hi.any() else 0.0,
            "low_prec_freq": lo.sum() / years, "low_prec_dur": np.mean(runs(lo)) if lo.any() else 0.0}

# ---- trained run ----------------------------------------------------------------------------------------------
def load_run(fam):
    run = RUNS[fam]; cfg = json.load(open(run / "cfg.json"))
    with h5py.File(run / "data/train/train_data.h5", "r") as f:
        train_basins = sorted(set(s.decode("ascii") for s in f["sample_2_basin"][:]))
    attr = load_attributes(str(run / "attributes.db"), train_basins, drop_lat_lon=True)  # 27 columns, training order
    model = Model(cfg["input_size_dyn"], cfg["hidden_size"], cfg["n_targets"], dropout=cfg["dropout"]).to(DEVICE)
    model.load_state_dict(torch.load(run / f"model_epoch{cfg['epochs']}.pt", map_location=DEVICE)); model.eval()
    assert cfg["input_size_dyn"] == len(DYN) + (2 if fam == "vt" else 0) + attr.shape[1], "input size mismatch"
    print(f"{fam}: {run.name}, epoch {cfg['epochs']}, {attr.shape[1]} attributes, {len(train_basins)} training basins")
    return {"family": fam, "cfg": cfg, "model": model, "attr_cols": list(attr.columns),
            "attr_means": attr.mean(), "attr_stds": attr.std(),
            "t_means": np.asarray(cfg["target_means"], np.float32), "t_stds": np.asarray(cfg["target_stds"], np.float32),
            "names": cfg["target_names"], "seq": cfg["seq_length"],
            "val": (pd.to_datetime(cfg["val_start"], format="%d%m%Y"), pd.to_datetime(cfg["val_end"], format="%d%m%Y"))}

def static_vector(run, stat_row):
    """stat_row: Series of CAMELS-named attributes for one unit; missing -> training mean (0 after standardisation)."""
    s = stat_row.reindex(run["attr_cols"])
    return ((s - run["attr_means"]) / run["attr_stds"]).fillna(0.0).to_numpy(np.float32)

@torch.no_grad()
def predict_windows(run, daily, x_stat, window=None):
    """daily: DataFrame indexed by date with DYN columns in NLDAS-equivalent units (full record).
    window: (start, end) of the window END dates; default = the run's validation period as in evaluate().
    Returns mu, sigma [n_windows, n_targets] in log(mm/day) and the window end dates."""
    L = run["seq"]; v0, v1 = window or run["val"]
    daily = daily.sort_index()
    df = daily[(daily.index >= v0 - pd.DateOffset(days=L - 1)) & (daily.index <= v1)]
    if len(df) < L + 1:
        raise ValueError(f"forcing covers {daily.index.min()}..{daily.index.max()}, need {v0 - pd.DateOffset(days=L - 1)}..{v1}")
    x = (df[DYN].to_numpy(np.float64) - SCALER["input_means"]) / SCALER["input_stds"]
    if run["family"] == "vt":
        x = np.concatenate([x, doy_features(df.index)], axis=1)
    xt = torch.tensor(x, dtype=torch.float32, device=DEVICE)
    wins = xt.unfold(0, L, 1).permute(0, 2, 1)                                        # [n, L, n_dyn] view
    sel = torch.arange(0, wins.shape[0], STRIDE, device=DEVICE)
    xs = torch.tensor(x_stat, device=DEVICE)
    mus, sgs = [], []
    for b in range(0, len(sel), BATCH):
        w = wins[sel[b:b + BATCH]]
        (mu, sg), _, _ = run["model"](torch.cat([w, xs.expand(w.shape[0], L, -1)], dim=-1))
        mus.append(mu.cpu()); sgs.append(sg.cpu())
    mu = torch.cat(mus).numpy() * run["t_stds"] + run["t_means"]
    sg = torch.cat(sgs).numpy() * run["t_stds"]
    return mu, sg, df.index[L - 1:][::STRIDE]

def summarise(run, mu, sg, dates, huc):
    fam, names = run["family"], run["names"]
    def row(m, s, extra):
        r = {"huc": huc, "level": LEVEL, **extra}
        for k, n in enumerate(names):
            r[f"{fam}_{n}_mu"], r[f"{fam}_{n}_sigma"] = m[k], s[k]
            r[f"{fam}_{n}_mmd"], r[f"{fam}_{n}_lo"], r[f"{fam}_{n}_hi"] = np.exp(m[k]), np.exp(m[k] - s[k]), np.exp(m[k] + s[k])
        return r
    if fam == "ft":
        return [row(mu.mean(0), sg.mean(0), {"n_windows": len(mu)})]
    doy = dates.dayofyear.to_numpy(); out = []
    for d in np.unique(doy):
        k = doy == d
        out.append(row(mu[k].mean(0), sg[k].mean(0), {"doy": int(d), "n_windows": int(k.sum())}))
    return out

# ---- main -----------------------------------------------------------------------------------------------------
def main():
    import gc
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corr = load_correction()
    print("Daymet->NLDAS: " + ("monthly linear correction from " + CORR_FILE.name if corr else "NO correction file - raw Daymet (run check_daymet_vs_nldas.py first)"))
    forc = daymet_to_nldas(load_forcing(), corr)
    stat = pd.read_csv(STATICS_CSV, dtype={"huc": str}).set_index("huc")
    clim = pd.DataFrame({h: climate_indices(g) for h, g in forc.groupby("huc", sort=False)}).T
    stat = stat.join(clim, how="inner")
    print(f"statics: {len(stat)} units x {stat.shape[1]} attributes computed; the rest take the training mean")
    # compact per-unit series (float32, DYN only) and drop the big frame
    series = {h: pd.DataFrame(g[DYN].to_numpy(np.float32), index=g["date"].to_numpy(), columns=DYN)
              for h, g in forc.groupby("huc", sort=False) if h in stat.index}
    del forc, clim; gc.collect()
    print(f"{len(series)} units ready")

    for fam in ["ft", "vt"]:
        out = OUT_DIR / f"huc_thresholds_{fam}_{LEVEL}.csv"
        done = set(pd.read_csv(out, dtype={"huc": str}, usecols=["huc"]).huc) if out.exists() else set()
        todo = [h for h in series if h not in done]
        print(f"{fam}: {len(done)} units already in {out.name}, {len(todo)} to do")
        if not todo:
            continue
        run = load_run(fam); rows = []
        for i, h in enumerate(todo, 1):
            mu, sg, dates = predict_windows(run, series[h], static_vector(run, stat.loc[h]))
            rows += summarise(run, mu, sg, dates, h)
            if i % CHECKPOINT_EVERY == 0 or i == len(todo):
                pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False); rows = []
                print(f"  {fam}: {len(done) + i}/{len(series)} units saved")
        print(f"{fam} done -> {out}")

if __name__ == "__main__":
    main()