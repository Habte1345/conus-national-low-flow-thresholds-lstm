"""Predict thresholds at every flowpath of the held-out basins of a fold.

Each basin is predicted by the model of the fold that held it out, so running all folds gives an
out-of-sample threshold for every flowpath of every gauged basin.

Windows end every K days within predict_period; each window yields the predictive mean and
spread (mu_t, sigma_t) of log(T + eps) for its last K days. Per flowpath they are summarized as a
Gaussian mixture over the windows
    mu = mean_t mu_t,  sigma^2 = mean_t sigma_t^2 + var_t mu_t
per day of year ('vt') or over all windows ('fq').

Output per basin: <run>/predictions/<basin>.npz
    flowpath, area_km2, names, mu_log, sigma_log  ((n_fp, 366, P) for 'vt', (n_fp, Q) for 'fq')
and <run>/predictions/<basin>_gauges.csv (thresholds at the outlet and interior gauge flowpaths).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .config import load_config, run_dir
from .features import Scaler, doy_features, static_features
from .model import ThresholdLSTM
from .thresholds import target_names


def load_run(rd, device):
    ck = torch.load(rd / "model.pt", map_location=device, weights_only=False)
    c = ck["config"]
    model = ThresholdLSTM(ck["n_in"], ck["n_out"], c["hidden_size"], c["dropout"], c["initial_forget_bias"],
                          monotone=(c["family"] == "vt")).to(device)
    model.load_state_dict(ck["model"]); model.eval()
    return model, Scaler(ck["scaler"]), ck


@torch.no_grad()
def predict_basin(cfg, model, scaler, f_npz, device):
    z = np.load(f_npz, allow_pickle=False)
    L, K, vt = int(cfg.seq_length), int(cfg.predict_last), cfg.family == "vt"
    dates = pd.DatetimeIndex(z["dates"].astype("datetime64[D]"))
    W = sp.csr_matrix((z["W_data"], z["W_indices"], z["W_indptr"]), shape=tuple(z["W_shape"]))
    Xd = z["divide_forcing"]; nd, nt, nv = Xd.shape
    Xd = Xd.reshape(nd, -1)
    S = scaler.stat(static_features(z["fp_static"], [str(n) for n in z["static_names"]], z["fp_area"]))
    D = doy_features(dates)
    p0, p1 = pd.Timestamp(cfg.predict_period[0]), pd.Timestamp(cfg.predict_period[1])
    ends = np.arange(L - 1, nt)
    ends = ends[(dates[ends] >= p0) & (dates[ends] <= p1)][::K]
    doy = dates.dayofyear.values - 1
    n_fp, n_out = W.shape[0], model.n_out
    shape = (n_fp, 366, n_out) if vt else (n_fp, n_out)
    MU, SG = np.full(shape, np.nan, np.float32), np.full(shape, np.nan, np.float32)

    for i in range(n_fp):
        x = np.asarray(W[i] @ Xd).reshape(nt, nv)
        bad = np.cumsum(np.r_[0, np.isnan(x).any(1).astype(int)])
        e_ok = ends[(bad[ends + 1] - bad[ends + 1 - L]) == 0]
        if len(e_ok) == 0:
            continue
        xz = scaler.dyn(x).astype(np.float32)
        if vt:
            xz = np.concatenate([xz, D], axis=1)
        xz = np.concatenate([xz, np.broadcast_to(S[i], (nt, S.shape[1]))], axis=1)
        xt = torch.from_numpy(xz).to(device)
        mus, sgs = [], []
        for j in range(0, len(e_ok), cfg.predict_batch):
            e = torch.as_tensor(e_ok[j:j + cfg.predict_batch], device=device)
            win = xt[(e[:, None] - L + 1) + torch.arange(L, device=device)[None, :]]       # (B, L, n_in)
            mu, sg = model(win, K)
            mus.append(mu.cpu().numpy()); sgs.append(sg.cpu().numpy())
        mu = np.concatenate(mus).reshape(-1, n_out); sg = np.concatenate(sgs).reshape(-1, n_out)
        mu, sg = scaler.y_inv(mu, sg)                                       # log(T + eps)
        days = (e_ok[:, None] - K + 1 + np.arange(K)[None, :]).reshape(-1)
        if vt:
            dd = doy[days]
            for d in np.unique(dd):
                m = dd == d
                MU[i, d] = mu[m].mean(0)
                SG[i, d] = np.sqrt((sg[m] ** 2).mean(0) + mu[m].var(0))
        else:
            MU[i] = mu.mean(0)
            SG[i] = np.sqrt((sg ** 2).mean(0) + mu.var(0))
    if vt:                                                                  # day 366 only in leap years
        miss = ~np.isfinite(MU[:, 365, 0])
        MU[miss, 365], SG[miss, 365] = MU[miss, 364], SG[miss, 364]
    return z, MU, SG


def gauge_table(cfg, z, MU, SG) -> pd.DataFrame:
    meta = pd.DataFrame(json.loads(str(z["target_meta"])))
    if meta.empty:
        return meta
    fpos = {int(v): i for i, v in enumerate(z["fp"])}
    names = target_names(cfg); eps = cfg.log_eps; rows = []
    obs = z["target_vt"] if cfg.family == "vt" else z["target_fq"]
    for j, r in meta.iterrows():
        i = fpos[int(r.flowpath)]
        if cfg.family == "vt":
            for d in range(366):
                for k, n in enumerate(names):
                    rows.append(dict(r, doy=d + 1, name=n, obs_mmd=obs[j, d, k], mu_log=MU[i, d, k], sigma_log=SG[i, d, k]))
        else:
            for k, n in enumerate(names):
                rows.append(dict(r, name=n, obs_mmd=obs[j, k], mu_log=MU[i, k], sigma_log=SG[i, k]))
    t = pd.DataFrame(rows)
    t["obs_log"] = np.log(np.clip(t.obs_mmd, 0, None) + eps)
    t["pred_mmd"] = np.clip(np.exp(t.mu_log) - eps, 0, None)
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--family", choices=["vt", "fq"])
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--basins", nargs="*", help="default: the basins held out by this fold")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, {"family": a.family, "fold": a.fold, "seed": a.seed})
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    rd = run_dir(cfg)
    model, scaler, ck = load_run(rd, device)
    basins = a.basins or ck["test_basins"]
    out = rd / "predictions"; out.mkdir(exist_ok=True)
    names = target_names(cfg)
    for k, b in enumerate(basins, 1):
        f = Path(cfg.cache_dir) / "basins" / f"{b}.npz"
        if not f.exists():
            continue
        z, MU, SG = predict_basin(cfg, model, scaler, f, device)
        np.savez_compressed(out / f"{b}.npz", flowpath=z["fp"], area_km2=z["fp_area"], names=np.array(names),
                            mu_log=MU, sigma_log=SG, log_eps=cfg.log_eps, family=cfg.family)
        gauge_table(cfg, z, MU, SG).to_csv(out / f"{b}_gauges.csv", index=False)
        print(f"[{k}/{len(basins)}] {b}: {len(z['fp'])} flowpaths", flush=True)


if __name__ == "__main__":
    main()
