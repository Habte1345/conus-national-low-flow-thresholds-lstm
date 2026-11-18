#!/bin/bash
#SBATCH --job-name=lf_prep
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/preprocess_%j.out
source ~/.bashrc
conda activate nh-dev
cd /bighome/hdagne1/conus-national-low-flow-thresholds-lstm
mkdir -p logs
python -u -m lowflow preprocess --config configs/default.yml
