#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J matu-pipeline
#SBATCH --chdir=/dl_scratch2/ege/ege/repos/MATU
#SBATCH --nodelist=adep1
#SBATCH --gres=gpu:1
#SBATCH --output=/dl_scratch2/ege/ege/repos/MATU/results/run_matu-%j.out
#SBATCH --error=/dl_scratch2/ege/ege/repos/MATU/results/run_matu-%j.err
#SBATCH --time=1-00:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/ege/envs/matu-uq/

# HF modelleri /home yerine dl_scratch2'ye insin
export HF_HOME=/dl_scratch2/ege/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /dl_scratch2/ege/ege/repos/MATU

echo "=== Step 04: Evaluate Reference (saniyeler) ==="
python quick_start/code/04_evaluate_reference_results.py --sample all

echo ""
echo "=== Step 05: Evaluate Baselines (saniyeler) ==="
python quick_start/code/05_evaluate_baselines.py --sample all

echo ""
echo "=== Step 01: Embed Logs (GPU, dakikalar) ==="
python quick_start/code/01_embed_reference_logs.py

echo ""
echo "=== Step 02: CP-2 (CPU, dakikalar-saat) ==="
python quick_start/code/02_run_cp2_from_generated_embeddings.py \
  --embedding-source generated \
  --max_rank 50

echo ""
echo "=== Step 03: Fit to Uncertainty ==="
python quick_start/code/03_fit_to_uncertainty_generated.py

echo ""
echo "=== Step 04: Evaluate Generated ==="
python quick_start/code/04_evaluate_generated_results.py

echo "MATU PIPELINE DONE!"
