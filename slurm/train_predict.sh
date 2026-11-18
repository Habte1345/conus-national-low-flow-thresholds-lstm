#!/bin/bash
#SBATCH --job-name=lf_fold
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
##SBATCH --constraint=l4      # uncomment with your cluster's feature name: torch cu130 has no V100 (CC 7.0) kernels
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/fold_%A_%a.out
# one array task per cross-validation fold; FAMILY=vt (default) or FAMILY=fq
source ~/.bashrc
conda activate nh-dev
cd /bighome/hdagne1/conus-national-low-flow-thresholds-lstm
mkdir -p logs
FAMILY=${FAMILY:-vt}
SEED=${SEED:-42}
python -u -m lowflow train   --config configs/default.yml --family $FAMILY --fold $SLURM_ARRAY_TASK_ID --seed $SEED
python -u -m lowflow predict --config configs/default.yml --family $FAMILY --fold $SLURM_ARRAY_TASK_ID --seed $SEED
