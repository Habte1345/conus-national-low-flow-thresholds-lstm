"""Train one fold.

Outlet basins are split into n_folds groups. Every gauge of a basin (outlet and interior) stays
in the basin's fold, so the held-out fold is spatially independent: neither its outlets nor its
interior gauges are seen in training. The model is fitted with the mean CRPS over the last K days
of each window and over all thresholds of the family.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from .config import load_config, run_dir
from .dataset import WindowSampler, load_targets
from .losses import crps_loss
from .model import ThresholdLSTM


def folds(cfg) -> dict:
    t = pd.read_csv(cfg.cache_dir / "targets.csv", dtype={"basin": str, "gauge": str})
    basins = np.array(sorted(t.basin.unique()))
    rng = np.random.default_rng(cfg.seed)
    perm = rng.permutation(basins)
    return {k: sorted(perm[k::cfg.n_folds].tolist()) for k in range(cfg.n_folds)}


def build_model(cfg, n_in, n_out):
    return ThresholdLSTM(n_in, n_out, cfg.hidden_size, cfg.dropout, cfg.initial_forget_bias,
                         monotone=(cfg.family == "vt"))


def evaluate_sampler(model, sampler, device, n_batches=20, B=256):
    model.eval(); tot = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = sampler.batch(B)
            mu, sg = model(torch.from_numpy(x).to(device), sampler.K)
            tot.append(crps_loss(mu, sg, torch.from_numpy(y).to(device)).item())
    model.train()
    return float(np.mean(tot))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--family", choices=["vt", "fq"])
    ap.add_argument("--fold", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--epochs", type=int)
    a = ap.parse_args(argv)
    cfg = load_config(a.config, {"family": a.family, "fold": a.fold, "seed": a.seed, "epochs": a.epochs})
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")

    split = folds(cfg)
    test_b = split[cfg.fold]
    train_b = [b for k, v in split.items() if k != cfg.fold for b in v]
    rd = run_dir(cfg); rd.mkdir(parents=True, exist_ok=True)

    tr_data = load_targets(cfg, train_b)
    if not len(tr_data["X"]):
        raise SystemExit("no cached training gauges - run preprocess first")
    tr = WindowSampler(tr_data, cfg, seed=cfg.seed)
    if not tr.ok.any():
        raise SystemExit(f"no training gauge has valid '{cfg.family}' targets (for 'fq', each gauge needs at least "
                         f"5 complete climate years in threshold_period)")
    te_data = load_targets(cfg, test_b)
    te = WindowSampler(te_data, cfg, scaler=tr.scaler, seed=cfg.seed + 1) if len(te_data["X"]) else None
    print(f"family {cfg.family} fold {cfg.fold}: {int(tr.ok.sum())} training gauges in {len(train_b)} basins, "
          f"{int(te.ok.sum()) if te else 0} test gauges in {len(test_b)} basins | inputs {tr.n_dyn}+{tr.n_stat}, "
          f"outputs {tr.n_out} | device {device}", flush=True)

    model = build_model(cfg, tr.n_dyn + tr.n_stat, tr.n_out).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    log = []
    for ep in range(1, cfg.epochs + 1):
        if ep in cfg.lr_drops:
            for g in opt.param_groups:
                g["lr"] = cfg.lr_drops[ep]
        model.train(); run = []
        for _ in range(cfg.batches_per_epoch):
            x, y = tr.batch(cfg.batch_size)
            x, y = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
            mu, sg = model(x, tr.K)
            loss = crps_loss(mu, sg, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            opt.step(); run.append(loss.item())
        rec = {"epoch": ep, "train_crps": float(np.mean(run)),
               "test_crps": evaluate_sampler(model, te, device) if te else np.nan}
        log.append(rec)
        print(f"epoch {ep:3d}  train CRPS {rec['train_crps']:.4f}  test CRPS {rec['test_crps']:.4f}", flush=True)

    torch.save({"model": model.state_dict(), "scaler": tr.scaler.state(), "n_in": tr.n_dyn + tr.n_stat,
                "n_out": tr.n_out, "config": {k: (str(v) if hasattr(v, "parts") else v) for k, v in cfg.items()},
                "train_basins": train_b, "test_basins": test_b}, rd / "model.pt")
    pd.DataFrame(log).to_csv(rd / "training_log.csv", index=False)
    json.dump({"train": train_b, "test": test_b}, open(rd / "split.json", "w"), indent=1)
    print(f"saved {rd / 'model.pt'}")


if __name__ == "__main__":
    main()
