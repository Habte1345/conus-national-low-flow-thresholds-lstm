"""
LSTM-FT / LSTM-VT / LSTM-FQ: national low-flow and hydrological-drought thresholds
across GAGES-II reference sites, trained with the CRPS (Gneiting & Raftery 2007).

Adapted from the accompanying code to:
Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. (2019).
Toward improved predictions in ungauged basins: Exploiting the power of machine learning.
Water Resources Research, 55. https://doi.org/10.1029/2019WR026065  (Apache-2.0)

Usage
-----
  python src/main.py train    --family ft --thresh_file data/thresholds.parquet --camels_root <root> --concat_static True --gpu 0
  python src/main.py train    --family vt ...
  python src/main.py train    --family fq ...
  python src/main.py evaluate --run_dir runs/run_XXXX_seedYYY --camels_root <root> --gpu 0 [--epoch 30]

Changes vs. the original main.py
--------------------------------
* --family {ft,vt,fq}, --thresh_file, --loss {crps,mse}, --epoch
* Model head: Linear(hidden, 2*n_targets) -> (mu, softplus sigma); forward returns
  ((mu, sigma), h_n, c_n)
* input_size_dyn and n_targets are read from the packed .h5, not hard-coded
* evaluate() writes per-basin (mu, sigma, obs, crps) and a family summary CSV
* seq_length 365, 30 epochs, lr 1e-3 -> 5e-4 -> 1e-4
"""

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.datasets import CamelsH5, CamelsTXT                      # noqa: E402
from data.datautils import (TARGET_NAMES, add_camels_attributes,   # noqa: E402
                            rescale_targets)
from Scripts.crpsloss import CRPSLoss, MaskedMSELoss, crps_gaussian  # noqa: E402
from Scripts.lstm import LSTM                                      # noqa: E402
from Scripts.utils import create_h5_files, get_basin_list          # noqa: E402

###########
# Globals #
###########

GLOBAL_SETTINGS = {
    'batch_size': 512,
    'clip_norm': True,
    'clip_value': 1,
    'dropout': 0.2,
    'epochs': 50,
    'hidden_size': 128,
    'initial_forget_gate_bias': 5,
    'log_interval': 100,
    'learning_rate': 1e-3,
    'seq_length': 365,
    'train_start': pd.to_datetime('01101980', format='%d%m%Y'),
    'train_end': pd.to_datetime('30092000', format='%d%m%Y'),
    'val_start': pd.to_datetime('01102000', format='%d%m%Y'),
    'val_end': pd.to_datetime('30092014', format='%d%m%Y'),
}

SIGMA_MIN = 1e-3

###############
# Prepare run #
###############


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate", "create_splits"])
    parser.add_argument('--camels_root', type=str, default='./data/basin_dataset_public_v1p2/',
                        help="Root directory of the CAMELS-format data set")
    parser.add_argument('--family', type=str, choices=list(TARGET_NAMES), default=None,
                        help="Threshold family: ft (fixed), vt (variable/DOY), fq (frequency)")
    parser.add_argument('--thresh_file', type=str, default=None,
                        help="Threshold table (.parquet/.csv): site_no, family, name, doy, value[mm/day]")
    parser.add_argument('--loss', type=str, choices=["crps", "mse"], default="crps")
    parser.add_argument('--seed', type=int, required=False, help="Random seed")
    parser.add_argument('--run_dir', type=str, help="For evaluation mode. Path to run directory.")
    parser.add_argument('--epoch', type=int, default=None,
                        help="For evaluation mode. Which model_epoch{N}.pt to load (default: last)")
    parser.add_argument('--gpu', type=int, default=-1, help="GPU id; -1 = cpu")
    parser.add_argument('--cache_data', type=str2bool, default=True)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--no_static', type=str2bool, default=False)
    parser.add_argument('--concat_static', type=str2bool, default=False)
    parser.add_argument('--n_splits', type=int, default=None)
    parser.add_argument('--basin_file', type=str, default=None)
    parser.add_argument('--split', type=int, default=None)
    parser.add_argument('--split_file', type=str, default=None)
    cfg = vars(parser.parse_args())

    if (cfg["mode"] in ["train", "create_splits"]) and (cfg["seed"] is None):
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

    if cfg["mode"] == "train":
        if cfg["family"] is None or cfg["thresh_file"] is None:
            raise ValueError("Training needs --family and --thresh_file")
        if cfg["no_static"] and cfg["concat_static"]:
            raise ValueError("--no_static and --concat_static are mutually exclusive")
        if not (cfg["no_static"] or cfg["concat_static"]):
            raise ValueError("This code path supports the standard LSTM only: "
                             "use --concat_static True (with statics) or --no_static True")

    if cfg["mode"] == "evaluate" and cfg["run_dir"] is None:
        raise ValueError("In evaluation mode a run directory (--run_dir) has to be specified")

    device = f"cuda:{cfg['gpu']}" if cfg["gpu"] >= 0 else "cpu"
    global DEVICE
    DEVICE = torch.device(device if torch.cuda.is_available() else "cpu")

    cfg.update(GLOBAL_SETTINGS)

    if cfg["mode"] == "train":
        for key, val in cfg.items():
            print(f"{key}: {val}")

    if cfg["camels_root"] is not None:
        cfg["camels_root"] = Path(cfg["camels_root"])
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])
    if cfg["thresh_file"] is not None:
        cfg["thresh_file"] = Path(cfg["thresh_file"]).resolve()
    return cfg


def _setup_run(cfg: Dict) -> Dict:
    now = datetime.now()
    run_name = (f'run_{now.day:02d}{now.month:02d}_{now.hour:02d}{now.minute:02d}'
                f'_{cfg["family"]}_seed{cfg["seed"]}')
    cfg['run_dir'] = Path(__file__).resolve().parents[1] / "runs" / run_name
    if cfg["run_dir"].is_dir():
        raise RuntimeError(f"There is already a folder at {cfg['run_dir']}")
    cfg["train_dir"] = cfg["run_dir"] / 'data' / 'train'
    cfg["train_dir"].mkdir(parents=True)
    cfg["val_dir"] = cfg["run_dir"] / 'data' / 'val'
    cfg["val_dir"].mkdir(parents=True)
    _dump_cfg(cfg)
    return cfg


def _dump_cfg(cfg: Dict):
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, (Path, PosixPath)):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            elif isinstance(val, np.ndarray):
                temp_cfg[key] = val.tolist()
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)


def _prepare_data(cfg: Dict, basins: List) -> Dict:
    cfg["db_path"] = str(cfg["run_dir"] / "attributes.db")
    add_camels_attributes(cfg["camels_root"], db_path=cfg["db_path"])

    cfg["train_file"] = cfg["train_dir"] / 'train_data.h5'
    create_h5_files(
        camels_root=cfg["camels_root"],
        out_file=cfg["train_file"],
        basins=basins,
        dates=[cfg["train_start"], cfg["train_end"]],
        family=cfg["family"],
        thresh_file=cfg["thresh_file"],
        with_basin_str=True,
        seq_length=cfg["seq_length"])
    return cfg


################
# Define Model #
################


class Model(nn.Module):
    """LSTM -> dropout -> linear head emitting (mu, sigma) for each threshold target."""

    def __init__(self,
                 input_size_dyn: int,
                 hidden_size: int,
                 n_targets: int,
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0):
        super(Model, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.hidden_size = hidden_size
        self.n_targets = n_targets

        self.lstm = LSTM(input_size=input_size_dyn,
                         hidden_size=hidden_size,
                         initial_forget_bias=initial_forget_bias)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 2 * n_targets)

    def forward(self, x_d: torch.Tensor):
        h_n, c_n = self.lstm(x_d)
        last_h = self.dropout(h_n[:, -1, :])
        out = self.fc(last_h)
        mu, s = out[:, :self.n_targets], out[:, self.n_targets:]
        sigma = F.softplus(s) + SIGMA_MIN
        return (mu, sigma), h_n, c_n


def _build_loss(name: str) -> nn.Module:
    return CRPSLoss() if name == "crps" else MaskedMSELoss()


###########################
# Train or evaluate model #
###########################


def train(cfg):
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    if cfg["split_file"] is not None:
        with Path(cfg["split_file"]).open('rb') as fp:
            splits = pickle.load(fp)
        basins = splits[cfg["split"]]["train"]
    else:
        basins = get_basin_list()

    cfg = _setup_run(cfg)
    cfg = _prepare_data(cfg=cfg, basins=basins)

    ds = CamelsH5(
        h5_file=cfg["train_file"],
        basins=basins,
        db_path=cfg["db_path"],
        concat_static=cfg["concat_static"],
        cache=cfg["cache_data"],
        no_static=cfg["no_static"])
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["num_workers"])

    # model dimensions from the data, not hard-coded
    n_static = 0 if cfg["no_static"] else len(ds.attribute_names)
    cfg["input_size_dyn"] = int(ds.n_dyn_features + n_static)
    cfg["n_targets"] = int(ds.n_targets)
    cfg["target_names"] = ds.target_names
    cfg["target_means"] = ds.target_means
    cfg["target_stds"] = ds.target_stds
    _dump_cfg(cfg)
    print(f"input_size_dyn={cfg['input_size_dyn']} (dyn {ds.n_dyn_features} + static {n_static}), "
          f"n_targets={cfg['n_targets']} {cfg['target_names']}")

    model = Model(
        input_size_dyn=cfg["input_size_dyn"],
        hidden_size=cfg["hidden_size"],
        n_targets=cfg["n_targets"],
        initial_forget_bias=cfg["initial_forget_gate_bias"],
        dropout=cfg["dropout"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_func = _build_loss(cfg["loss"])

    learning_rates = {11: 5e-4, 21: 1e-4}

    for epoch in range(1, cfg["epochs"] + 1):
        if epoch in learning_rates:
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rates[epoch]

        train_epoch(model, optimizer, loss_func, loader, cfg, epoch)

        model_path = cfg["run_dir"] / f"model_epoch{epoch}.pt"
        torch.save(model.state_dict(), str(model_path))


def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loss_func: nn.Module,
                loader: DataLoader, cfg: Dict, epoch: int):
    model.train()
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epoch}')

    running = []
    for data in pbar:
        optimizer.zero_grad()

        x, y, _q_stds = data
        x, y = x.to(DEVICE), y.to(DEVICE)
        (mu, sigma), _, _ = model(x)

        loss = loss_func(mu, sigma, y)
        loss.backward()

        if cfg["clip_norm"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"])

        optimizer.step()
        running.append(loss.item())
        pbar.set_postfix_str(f"{cfg['loss'].upper()}: {loss.item():.5f}")

    print(f"# Epoch {epoch}: mean {cfg['loss'].upper()} = {np.mean(running):.5f}")


def evaluate(user_cfg: Dict):
    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    family = run_cfg["family"]
    thresh_file = user_cfg["thresh_file"] or Path(run_cfg["thresh_file"])
    target_names = run_cfg["target_names"]
    means = np.asarray(run_cfg["target_means"], dtype=np.float32)
    stds = np.asarray(run_cfg["target_stds"], dtype=np.float32)

    if user_cfg["split_file"] is not None:
        with Path(user_cfg["split_file"]).open('rb') as fp:
            splits = pickle.load(fp)
        basins = splits[run_cfg["split"]]["test"]
    else:
        basins = get_basin_list()

    # attribute means/stds from the TRAINING basins (read back from the packed .h5)
    train_file = user_cfg["run_dir"] / "data/train/train_data.h5"
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    with h5py.File(train_file, "r") as f:
        train_basins = sorted(set(s.decode("ascii") for s in f["sample_2_basin"][:]))
    ds_train = CamelsH5(h5_file=train_file, db_path=db_path, basins=train_basins,
                        concat_static=run_cfg["concat_static"], no_static=run_cfg["no_static"])
    attr_means = ds_train.get_attribute_means()
    attr_stds = ds_train.get_attribute_stds()

    model = Model(
        input_size_dyn=run_cfg["input_size_dyn"],
        hidden_size=run_cfg["hidden_size"],
        n_targets=run_cfg["n_targets"],
        dropout=run_cfg["dropout"]).to(DEVICE)

    epoch = user_cfg["epoch"] or run_cfg["epochs"]
    weight_file = user_cfg["run_dir"] / f'model_epoch{epoch}.pt'
    model.load_state_dict(torch.load(weight_file, map_location=DEVICE))
    print(f"Loaded {weight_file}")

    val_start = pd.to_datetime(run_cfg["val_start"], format="%d%m%Y")
    val_end = pd.to_datetime(run_cfg["val_end"], format="%d%m%Y")

    results = {}
    for basin in tqdm(basins, file=sys.stdout):
        ds_test = CamelsTXT(
            camels_root=user_cfg["camels_root"],
            basin=basin,
            dates=[val_start, val_end],
            is_train=False,
            family=family,
            thresh_file=thresh_file,
            seq_length=run_cfg["seq_length"],
            with_attributes=not run_cfg["no_static"],
            attribute_means=attr_means,
            attribute_stds=attr_stds,
            concat_static=run_cfg["concat_static"],
            db_path=db_path)
        loader = DataLoader(ds_test, batch_size=1024, shuffle=False,
                            num_workers=min(4, user_cfg["num_workers"]))

        mu, sigma, obs = evaluate_basin(model, loader)          # standardised space
        mu, sigma = rescale_targets(mu, sigma, means, stds)     # -> log(mm/day)
        crps = crps_gaussian(torch.from_numpy(mu), torch.from_numpy(sigma),
                             torch.from_numpy(obs)).numpy()

        n = min(len(mu), len(ds_test.sample_dates))
        idx = ds_test.sample_dates[:n]
        cols = {}
        for j, name in enumerate(target_names):
            cols[f"obs_{name}"] = obs[:n, j]
            cols[f"mu_{name}"] = mu[:n, j]
            cols[f"sigma_{name}"] = sigma[:n, j]
            cols[f"crps_{name}"] = crps[:n, j]
        results[basin] = pd.DataFrame(cols, index=idx)

    _store_results(user_cfg, run_cfg, results, target_names)


def evaluate_basin(model: nn.Module, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    mus, sigmas, obs = [], [], []
    with torch.no_grad():
        for data in loader:
            if len(data) == 3:
                x, _attr, y = data
            else:
                x, y = data
            x = x.to(DEVICE)
            (mu, sigma), _, _ = model(x)
            mus.append(mu.cpu())
            sigmas.append(sigma.cpu())
            obs.append(y)
    return (torch.cat(mus).numpy(), torch.cat(sigmas).numpy(), torch.cat(obs).numpy())


def _store_results(user_cfg: Dict, run_cfg: Dict, results: Dict, target_names: List[str]):
    family = run_cfg["family"]
    tag = "lstm_no_static" if run_cfg["no_static"] else "lstm"
    stem = f"{family}_{tag}_seed{run_cfg['seed']}"

    # 1) full per-basin daily frame (obs / mu / sigma / crps per target), log(mm/day)
    pkl = user_cfg["run_dir"] / f"{stem}.p"
    with pkl.open('wb') as fp:
        pickle.dump(results, fp)

    # 2) family summary — the national threshold table
    rows = []
    for basin, df in results.items():
        if family == "vt":
            g = df.groupby(df.index.dayofyear)
            for doy, gdf in g:
                for name in target_names:
                    rows.append(_summary_row(basin, name, gdf, doy=int(doy)))
        else:
            for name in target_names:
                rows.append(_summary_row(basin, name, df))
    summary = pd.DataFrame(rows)
    csv = user_cfg["run_dir"] / f"{stem}_summary.csv"
    summary.to_csv(csv, index=False)

    overall = summary.groupby("name")["crps_log"].mean()
    print(f"Stored {pkl.name} and {csv.name}")
    print("Mean CRPS (log mm/day) per target:\n" + overall.to_string())


def _summary_row(basin: str, name: str, df: pd.DataFrame, doy: int = None) -> Dict:
    obs_log = float(np.nanmean(df[f"obs_{name}"]))
    mu_log = float(np.nanmean(df[f"mu_{name}"]))
    sigma_log = float(np.nanmean(df[f"sigma_{name}"]))
    row = {"site_no": basin, "name": name}
    if doy is not None:
        row["doy"] = doy
    row.update({
        "obs_log": obs_log,
        "mu_log": mu_log,
        "sigma_log": sigma_log,
        "obs_mmd": float(np.exp(obs_log)),
        "mu_mmd": float(np.exp(mu_log)),          # median of the lognormal predictive dist.
        "crps_log": float(np.nanmean(df[f"crps_{name}"])),
        "n_days": int(len(df)),
    })
    return row


####################
# Cross Validation #
####################


def create_splits(cfg: dict):
    """Random k-fold splits over basins (unchanged from the original)."""
    output_file = (Path(__file__).resolve().parents[1] / f'data/kfold_splits_seed{cfg["seed"]}.p')
    if output_file.is_file():
        raise RuntimeError(f"File '{output_file}' already exists.")

    np.random.seed(cfg["seed"])

    if cfg["basin_file"] is not None:
        if not Path(cfg["basin_file"]).is_file():
            raise FileNotFoundError(f"Not file found at {cfg['basin_file']}")
        with open(cfg["basin_file"], 'r') as fp:
            basins = [b.strip() for b in fp.readlines() if b.strip()]
    else:
        basins = get_basin_list()

    kfold = KFold(n_splits=cfg["n_splits"], shuffle=True, random_state=cfg["seed"])
    splits = defaultdict(dict)
    for split, (train_idx, test_idx) in enumerate(kfold.split(basins)):
        splits[split] = {'train': [basins[i] for i in train_idx],
                         'test': [basins[i] for i in test_idx]}

    with output_file.open('wb') as fp:
        pickle.dump(splits, fp)
    print(f"Stored dictionary with basin splits at {output_file}")


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)
