"""
Adapted from Kratzert et al. (2019), Apache-2.0.

Changes vs. the original utils.py
---------------------------------
create_h5_files
  * new args `family`, `thresh_file` (passed to CamelsTXT).
  * input width = n_dynamic_features(family) (5, or 7 for VT), target width = n_targets.
  * after all basins are packed, targets are standardised per column over ALL training
    samples (nan-aware) and the scaler is stored in the file:
        target_means, target_stds, target_names
get_basin_list unchanged.
"""
import sys
from pathlib import Path, PosixPath
from typing import List

import h5py
import numpy as np
from tqdm import tqdm

from data.datasets import CamelsTXT
from data.datautils import TARGET_NAMES, n_dynamic_features


def create_h5_files(camels_root: PosixPath,
                    out_file: PosixPath,
                    basins: List,
                    dates: List,
                    family: str,
                    thresh_file,
                    with_basin_str: bool = True,
                    seq_length: int = 365):
    """Pack training samples of all basins into one .h5 file."""
    if out_file.is_file():
        raise FileExistsError(f"File already exists at {out_file}")

    n_feat = n_dynamic_features(family)
    names = TARGET_NAMES[family]
    n_targets = len(names)

    with h5py.File(out_file, 'w') as out_f:
        input_data = out_f.create_dataset(
            'input_data',
            shape=(0, seq_length, n_feat),
            maxshape=(None, seq_length, n_feat),
            chunks=True,
            dtype=np.float32,
            compression='gzip')
        target_data = out_f.create_dataset(
            'target_data',
            shape=(0, n_targets),
            maxshape=(None, n_targets),
            chunks=True,
            dtype=np.float32,
            compression='gzip')
        q_stds = out_f.create_dataset(
            'q_stds',
            shape=(0, 1),
            maxshape=(None, 1),
            dtype=np.float32,
            compression='gzip',
            chunks=True)
        if with_basin_str:
            sample_2_basin = out_f.create_dataset(
                'sample_2_basin',
                shape=(0,),
                maxshape=(None,),
                dtype="S10",
                compression='gzip',
                chunks=True)

        for basin in tqdm(basins, file=sys.stdout):
            dataset = CamelsTXT(
                camels_root=camels_root,
                basin=basin,
                is_train=True,
                family=family,
                thresh_file=thresh_file,
                seq_length=seq_length,
                dates=dates)

            num_samples = len(dataset)
            total_samples = input_data.shape[0] + num_samples

            input_data.resize((total_samples, seq_length, n_feat))
            target_data.resize((total_samples, n_targets))
            input_data[-num_samples:, :, :] = dataset.x
            target_data[-num_samples:, :] = dataset.y

            q_stds.resize((total_samples, 1))
            q_stds[-num_samples:, :] = np.full((num_samples, 1), dataset.q_std, dtype=np.float32)

            if with_basin_str:
                sample_2_basin.resize((total_samples,))
                sample_2_basin[-num_samples:] = np.array(
                    [basin.encode("ascii", "ignore")] * num_samples)

            out_f.flush()

        # ---- global per-target standardisation of the log-thresholds -------------------
        y_all = target_data[:]
        means = np.nanmean(y_all, axis=0)
        stds = np.nanstd(y_all, axis=0)
        stds = np.where(stds < 1e-6, 1.0, stds)
        target_data[...] = ((y_all - means) / stds).astype(np.float32)

        out_f.create_dataset('target_means', data=means.astype(np.float32))
        out_f.create_dataset('target_stds', data=stds.astype(np.float32))
        out_f.create_dataset('target_names', data=np.array([n.encode("ascii") for n in names]))
        out_f.attrs['family'] = family
        out_f.flush()

    print(f"Packed {len(basins)} basins → {out_file}  "
          f"(targets: {names}; means={np.round(means, 3)}, stds={np.round(stds, 3)})")


def get_basin_list() -> List:
    """Read list of basins from data/basin_list.txt."""
    basin_file = Path(__file__).absolute().parent.parent / "data/basin_list.txt"
    with basin_file.open('r') as fp:
        basins = fp.readlines()
    return [basin.strip() for basin in basins if basin.strip()]
