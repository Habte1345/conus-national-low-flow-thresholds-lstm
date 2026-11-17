#!/bin/bash
#SBATCH --job-name=LSTM
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/hdagne1/conus-national-low-flow-thresholds-lstm/logs/train_ft_seed200_%j.out
#SBATCH --error=/scratch/hdagne1/conus-national-low-flow-thresholds-lstm/logs/train_ft_seed200_%j.err
source ~/.bashrc
conda activate nh-dev
cd /scratch/hdagne1/conus-national-low-flow-thresholds-lstm
mkdir -p logs runs

echo "job $SLURM_JOB_ID on $(hostname)  start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>/dev/null || echo "no GPU visible"

python src/main.py train --family ft --thresh_file data/thresholds.parquet \
  --camels_root data/CAMELS_US/ --concat_static True --gpu 0 --seed 200 --num_workers 8

echo "end $(date)  exit $?"