#!/bin/bash
#SBATCH --job-name=LSTM_eval
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/hdagne1/conus-national-low-flow-thresholds-lstm/logs/eval_ft_seed200_%j.out
#SBATCH --error=/scratch/hdagne1/conus-national-low-flow-thresholds-lstm/logs/eval_ft_seed200_%j.err
source ~/.bashrc
conda activate nh-dev
cd /scratch/hdagne1/conus-national-low-flow-thresholds-lstm
mkdir -p logs

# run directory: pass it as the first argument, or the newest ft run is used
RUN_DIR=${1:-$(ls -td runs/run_*_ft_seed200 | head -1)}
echo "job $SLURM_JOB_ID on $(hostname)  start $(date)"
echo "evaluating $RUN_DIR"
ls "$RUN_DIR"/model_epoch*.pt | tail -1

python src/main.py evaluate --run_dir "$RUN_DIR" \
  --camels_root data/CAMELS_US/ --gpu 0 --num_workers 8

echo "end $(date)  exit $?"
ls -lh "$RUN_DIR"/*_summary.csv "$RUN_DIR"/*.p 2>/dev/null