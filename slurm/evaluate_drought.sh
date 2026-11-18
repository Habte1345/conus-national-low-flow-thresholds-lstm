#!/bin/bash
#SBATCH --job-name=lf_eval
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_drought_%j.out
# run after all folds of train_predict.sh have finished
source ~/.bashrc
conda activate nh-dev
cd /bighome/hdagne1/conus-national-low-flow-thresholds-lstm
mkdir -p logs
SCEN=${SCEN:-all}
python -u -m lowflow evaluate    --config configs/default.yml --family vt
python -u -m lowflow evaluate    --config configs/default.yml --family fq
python -u -m lowflow troute_flow --config configs/default.yml --scenario $SCEN --out_dir data/sim_flow/neuralngen_$SCEN
python -u -m lowflow drought     --config configs/default.yml --flow_dir data/sim_flow/neuralngen_$SCEN --mode mapped \
       --period 2018-01-01 2023-01-31 --save_classes
