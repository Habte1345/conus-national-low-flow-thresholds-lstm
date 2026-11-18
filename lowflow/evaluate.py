"""Out-of-sample skill at gauges, pooled over all folds of one family.

Per gauge and threshold (day-of-year thresholds are first compared day by day, then averaged):
  crps_log    CRPS of N(mu, sigma^2) against log(T_obs + eps)
  bias_log10  mean log10(T_pred / T_obs)
  abs_log10   mean |log10(T_pred / T_obs)|
  cov90       share of days with log(T_obs + eps) inside mu +/- 1.645 sigma
The summary separates outlet gauges from interior gauges, which test the within-basin
distribution of thresholds on flowpaths never seen in training.

Output: <runs_dir>/evaluation_<family>_seed<seed>.csv and ..._summary.csv
"""
import argparse
import glob

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import load_config


def crps_np(mu, sg, y):
    z = (y - mu) / sg
    return sg * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--family", choices=["vt", "fq"])
    ap.add_argument("--seed", type=int)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, {"family": a.family, "seed": a.seed})
    files = glob.glob(str(cfg.runs_dir / f"{cfg.family}_fold*_seed{cfg.seed}" / "predictions" / "*_gauges.csv"))
    t = pd.concat([pd.read_csv(f, dtype={"basin": str, "gauge": str}) for f in files], ignore_index=True)
    t = t.dropna(subset=["obs_log", "mu_log", "sigma_log"])
    eps = cfg.log_eps
    t["crps_log"] = crps_np(t.mu_log, t.sigma_log, t.obs_log)
    r = (np.exp(t.mu_log)) / (np.clip(t.obs_mmd, 0, None) + eps)
    t["err_log10"] = np.log10(r)
    t["inside90"] = (np.abs(t.obs_log - t.mu_log) <= 1.645 * t.sigma_log).astype(float)
    keys = ["basin", "gauge", "kind", "flowpath", "area_km2", "name"]
    g = t.groupby(keys).agg(crps_log=("crps_log", "mean"), bias_log10=("err_log10", "mean"),
                            abs_log10=("err_log10", lambda x: np.abs(x).mean()), cov90=("inside90", "mean"),
                            obs_mmd=("obs_mmd", "mean"), pred_mmd=("pred_mmd", "mean")).reset_index()
    out = cfg.runs_dir / f"evaluation_{cfg.family}_seed{cfg.seed}.csv"
    g.to_csv(out, index=False)
    s = g.groupby(["kind", "name"]).agg(n=("gauge", "nunique"), crps_log=("crps_log", "median"),
                                         bias_log10=("bias_log10", "median"), abs_log10=("abs_log10", "median"),
                                         cov90=("cov90", "median")).reset_index()
    s.to_csv(str(out).replace(".csv", "_summary.csv"), index=False)
    print(s.round(3).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
