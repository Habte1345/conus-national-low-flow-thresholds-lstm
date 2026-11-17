#!/bin/bash
#SBATCH --job-name=huc8_thresholds
#SBATCH --account=mmmekonnen_grp
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/bighome/hdagne1/conus-national-low-flow-thresholds-lstm/logs/huc8_thresholds_%j.out
source ~/.bashrc
conda activate nh-dev
cd /bighome/hdagne1/conus-national-low-flow-thresholds-lstm
mkdir -p logs
python Scripts/predict_huc8.py