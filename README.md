# conus-national-low-flow-thresholds-lstm

Low-flow and hydrological drought thresholds at every flowpath of the NextGen hydrofabric, learned
from gauged outlets and interior gauges with a probabilistic LSTM, and the streamflow drought they
imply at every flowpath.

Version 2 replaces the HUC8 workflow of version 1 (tag `v1-huc8`). Thresholds are now defined on
the hydrofabric v2.2 network used by NextGen, at real drainage points with a known upstream area.

## Method

**Unit.** A flowpath *f* of the hydrofabric with upstream divides *U(f)* and drainage area
*A_f* = Σ_{c∈U(f)} *A_c*. Divide `cat-N` drains to flowpath `wb-N`; the network follows
`flowpaths.toid` (nexus) and `nexus.toid` (flowpath).

**Inputs.** Daily AORC forcing and the 33 hydrofabric attributes, both area-weighted over *U(f)*
(weights *A_c / A_f*), with circular attributes averaged as unit vectors, plus log10 *A_f*. For
day-of-year thresholds the sine and cosine of the day of year are added.

**Targets.** Observed thresholds at the outlet gauge and at every interior USGS gauge of a basin,
from daily flow converted to depth over the drainage area of the gauge flowpath.

- `vt`, variable thresholds. Day-of-year percentiles P5 to P30 (Weibull plotting position),
  pooling all days within ±15 days of each calendar day, smoothed with a 7-day circular mean.
- `fq`, frequency thresholds. 1Q2, 1Q10, 7Q2, 7Q5, 7Q10, 30Q2, 30Q5, 30Q10 from annual minima
  of the n-day mean flow per climate year (April to March), log-Pearson type III, with the
  conditional-probability adjustment for zero minima.

**Model.** One LSTM shared by all flowpaths reads 365 days of inputs and returns, for each of the
last *K* days, a Gaussian predictive distribution of log(*T* + ε) for every threshold. For `vt` the
means are built as cumulative positive increments, so P5 ≤ P10 ≤ … ≤ P30 holds at every flowpath
and day.

**Loss.** The closed-form CRPS of the Gaussian predictive distribution (Gneiting and Raftery,
2007), averaged over thresholds, days and gauges. Outlets and interior gauges are scored together,
so the observations inside each basin constrain how thresholds vary within it.

**Validation.** Cross-validation over outlet basins. Every gauge of a basin, outlet and interior,
stays in the basin's fold, so the held-out gauges and all flowpaths of a held-out basin are never
seen in training. Interior gauges test the within-basin distribution of thresholds directly.

**Summary per flowpath.** Window predictions are pooled as a Gaussian mixture,
μ = mean μ_t and σ² = mean σ_t² + var μ_t, per day of year (`vt`) or over all windows (`fq`).

**Drought.** Simulated daily flow at every flowpath (t-route output of the NeuralNGEN runs, or
NWM) is compared with the thresholds. In the default `mapped` mode the simulated day-of-year
percentiles are mapped onto the learned thresholds by a monotone piecewise-linear transform in log
space, so drought timing follows the simulated percentiles while deficit volumes are anchored to
the observed thresholds. Classes follow threshold bands. None Q ≥ P30, D0 P25–P30, D1 P20–P25,
D2 P10–P20, D3 P5–P10, D4 Q < P5. Events are runs below a threshold, with gaps of up to 5 days
merged and a minimum length of 7 days.

## Data layout (NGIAB subsets of the NeuralNGEN study)

```
/bighome/hdagne1/Data/CAMELS_US_ngiab_full/gage-<id>/config/<id>_subset.gpkg   divides, divide-attributes, flowpaths, nexus
/bighome/hdagne1/Data/CAMELS_US_ngiab_full/gage-<id>/forcings/forcings.nc      ids, Time, hourly AORC variables
/bighome/hdagne1/Data/CAMELS_US_ngiab_full/gage-<id>/outputs/troute_<s>_stream/ t-route output (drought step)
/bighome/hdagne1/Data/usgs_streamflow_ref/<id>.parquet                          outlet flow, flow_m3s, daily
/bighome/hdagne1/NeuralNGEN/interior_gauges.csv                                 outlet_gage, interior_gage, flowpath
/bighome/hdagne1/NeuralNGEN/obs_cache_interior/<id>.csv                         interior flow, datetime, Q_m3s
```

## Workflow

```bash
python -m lowflow preprocess  --config configs/default.yml                     # cache per basin + targets.csv
python -m lowflow train       --config configs/default.yml --family vt --fold 0
python -m lowflow predict     --config configs/default.yml --family vt --fold 0  # held-out basins of the fold
python -m lowflow evaluate    --config configs/default.yml --family vt           # pooled over folds
python -m lowflow troute_flow --config configs/default.yml --scenario all --out_dir data/sim_flow/neuralngen_all
python -m lowflow drought     --config configs/default.yml --flow_dir data/sim_flow/neuralngen_all --mode mapped
```

On the HPC: `sbatch slurm/preprocess.sh`, then `sbatch slurm/train_predict.sh` (array over folds;
`FAMILY=fq sbatch ...` for frequency thresholds), then `sbatch slurm/evaluate_drought.sh`.

Smoke test on a mock NGIAB layout: `pytest -q tests/test_pipeline.py`.

## Outputs

| File | Content |
|---|---|
| `data/cache/basins/<id>.npz` | daily divide forcing, area weights, flowpath attributes, gauge targets |
| `data/cache/targets.csv` | gauges with valid targets (outlet or interior, flowpath, area, record length) |
| `runs/<family>_fold<k>_seed<s>/model.pt` | model, scalers, train and test basins |
| `runs/<family>_fold<k>_seed<s>/predictions/<id>.npz` | μ and σ of log(T + ε) at every flowpath |
| `runs/<family>_fold<k>_seed<s>/predictions/<id>_gauges.csv` | predicted and observed thresholds at gauges |
| `runs/evaluation_<family>_seed<s>[_summary].csv` | CRPS, bias, absolute log error, 90% coverage |
| `runs/drought_<mode>_seed<s>/<id>_summary.csv` | events, durations, deficits, onset per flowpath |
| `runs/drought_<mode>_seed<s>/<id>_classes.parquet` | daily drought class per flowpath |

## Notes

- `threshold_period` must lie inside the NGIAB forcing record. Thresholds need long records;
  frequency thresholds need at least 5 complete climate years per gauge.
- Interior gauges are not all reference gauges. List regulated ones in `interior_exclude_file`.
- The drought step is only as long as the simulated flow; the 2018 to 2023 NeuralNGEN evaluation
  runs give five years, which is short for the simulated percentiles in `mapped` and `model` modes.
