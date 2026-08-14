#!/bin/bash
#SBATCH --job-name=eigv_calc
#SBATCH --output=eigv_calc_%j.out
#SBATCH --error=eigv_calc_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --chdir=/dl_scratch2/ege/ege/repos/MATU
#SBATCH --time=12:00:00

# Ortami aktif et
source ~/.bashrc
conda activate matu-uq

# Scripti calistir
python baselines/run_eigv_analysis.py
