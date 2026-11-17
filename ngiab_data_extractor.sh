#!/bin/bash
#SBATCH --job-name=NGIAB
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/bighome/hdagne1/Docker/NeuralNGEN/runs/logs/forcing_test_%j.out
#SBATCH --error=/bighome/hdagne1/Docker/NeuralNGEN/runs/logs/forcing_test_%j.err

source ~/.bashrc
conda activate nh-dev

python -m ngiab_data_cli \
  -i 01022500 -g \
  -s -f \
  --start_date 2020-01-01 --end_date 2024-12-31 \
  --source aorc \
  --subset_type catchment \
  --output_root /bighome/hdagne1/Docker/Data/CAMELS_US_ngiab_full \
  -o gage-01022500

echo "Done."