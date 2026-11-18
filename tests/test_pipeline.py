"""End-to-end smoke test on a mock NGIAB layout: preprocess -> train -> predict -> evaluate -> drought.

    pytest -q tests/test_pipeline.py        (about a minute on CPU)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_mock_ngiab import make                                           # noqa: E402
from lowflow import drought, evaluate, predict, preprocess, train         # noqa: E402


def write_cfg(root):
    cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs/default.yml"))
    cfg.update(
        gauges_root=str(root / "Data/CAMELS_US_ngiab_full"), streamflow_dir=str(root / "Data/usgs_streamflow_ref"),
        basin_file=str(root / "NeuralNGEN/basin_lists/basins.txt"), interior_file=str(root / "NeuralNGEN/interior_gauges.csv"),
        interior_obs_dir=str(root / "NeuralNGEN/obs_cache_interior"), cache_dir=str(root / "cache"), runs_dir=str(root / "runs"),
        threshold_period=["2000-04-01", "2006-03-31"], predict_period=["2000-10-01", "2006-03-31"], min_years=5,
        seq_length=90, predict_last=10, hidden_size=16, epochs=2, batches_per_epoch=10, batch_size=16, n_folds=3,
        device="cpu", lr_drops={}, predict_batch=64)
    p = root / "test.yml"; yaml.safe_dump(cfg, open(p, "w"))
    return str(p)


def test_pipeline(tmp_path):
    root = make(tmp_path, n_basins=6, n_fp=10, end="2006-12-31")
    c = write_cfg(root)
    preprocess.main(["--config", c])
    t = pd.read_csv(root / "cache/targets.csv", dtype=str)
    assert (t.kind == "outlet").sum() == 6 and (t.kind == "interior").sum() == 12
    for fam in ("vt", "fq"):
        for k in range(3):
            train.main(["--config", c, "--family", fam, "--fold", str(k)])
            predict.main(["--config", c, "--family", fam, "--fold", str(k)])
        evaluate.main(["--config", c, "--family", fam])
    p = np.load(next((root / "runs").glob("vt_fold0_seed42/predictions/*.npz")))
    mu = p["mu_log"]
    assert np.isfinite(mu).all() and (np.diff(mu, axis=-1) >= 0).all()      # P5 <= ... <= P30 everywhere

    # simulated flow = observed outlet flow scaled to every flowpath by area (stand-in for t-route output)
    fd = root / "flow"; fd.mkdir()
    for f in (root / "runs").glob("vt_fold*_seed42/predictions/*.npz"):
        z = np.load(f)
        q = pd.read_parquet(root / "Data/usgs_streamflow_ref" / f"{f.stem}.parquet")["flow_m3s"]
        A = z["area_km2"]
        pd.DataFrame({f"wb-{int(fp)}": 0.8 * q.values * a / A.max() for fp, a in zip(z["flowpath"], A)},
                     index=q.index).to_parquet(fd / f"{f.stem}.parquet")
    drought.main(["--config", c, "--flow_dir", str(fd), "--save_classes"])
    s = pd.concat([pd.read_csv(f) for f in (root / "runs/drought_mapped_seed42").glob("*_summary.csv")])
    assert len(s) and s.drought_days_per_year.between(0, 366).all()
