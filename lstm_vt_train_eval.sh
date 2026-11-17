#!/bin/bash
#SBATCH --job-name=lstm_vt
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=30:00:00
#SBATCH --output=/scratch/hdagne1/conus-national-low-flow-thresholds-lstm/logs/vt_seed200_%j.out
#SBATCH --error=/scratch/hdagne1/conus-national-low-flow-thresholds-lstm/logs/vt_seed200_%j.err

source ~/.bashrc
conda activate nh-dev
cd /scratch/hdagne1/conus-national-low-flow-thresholds-lstm
mkdir -p logs runs

echo "job $SLURM_JOB_ID on $(hostname)  start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "no GPU visible"

# ---- train --------------------------------------------------------------------------------
python src/main.py train --family vt --thresh_file data/thresholds.parquet \
  --camels_root data/CAMELS_US/ --concat_static True --gpu 0 --seed 200 --num_workers 0
status=$?
echo "training end $(date)  exit $status"
if [ $status -ne 0 ]; then echo "training failed, not evaluating"; exit $status; fi

# ---- evaluate the run just created ---------------------------------------------------------
RUN_DIR=$(ls -td runs/run_*_vt_seed200 | head -1)
echo "evaluating $RUN_DIR"
ls "$RUN_DIR"/model_epoch*.pt | tail -1

python src/main.py evaluate --run_dir "$RUN_DIR" \
  --camels_root data/CAMELS_US/ --gpu 0 --num_workers 0

echo "evaluation end $(date)  exit $?"
ls -lh "$RUN_DIR"/*_summary.csv "$RUN_DIR"/*.p 2>/dev/null