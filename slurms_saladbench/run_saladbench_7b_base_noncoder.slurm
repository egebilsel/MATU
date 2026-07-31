#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J sb-7b_base_nc
#SBATCH --chdir=/dl_scratch2/ege/ege/repos/MATU
#SBATCH --nodelist=adep1
#SBATCH --gres=gpu:1
#SBATCH --output=/dl_scratch2/ege/ege/repos/MATU/results/saladbench_7b_base_noncoder-%j.out
#SBATCH --error=/dl_scratch2/ege/ege/repos/MATU/results/saladbench_7b_base_noncoder-%j.err
#SBATCH --time=3-00:00:00

# 1. Environment ve Cache Ayarları
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/ege/envs/matu-uq/

export HF_HOME=/dl_scratch2/ege/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /dl_scratch2/ege/ege/repos/MATU

# Çıktı klasörlerini oluştur
mkdir -p quick_start/saladbench_7b_base_noncoder/embeddings
mkdir -p quick_start/saladbench_7b_base_noncoder/results
mkdir -p quick_start/data/saladbench

echo "=== Step 00: Generate SaladBench Logs (GPU) ==="
python examples/generate_logs_saladbench.py \
  --max_items 100 \
  --model "Qwen/Qwen2.5-7B-Instruct" \
  --max_items 100 \
  --runs 10 \
  --seed 42 \
  --out quick_start/saladbench_7b_base_noncoder/conversation_logs_saladbench.json

echo ""
echo "=== Step 01: Embed Logs (GPU) ==="
python -m matu.embed_logs \
  --logs quick_start/saladbench_7b_base_noncoder/conversation_logs_saladbench.json \
  --out_dir quick_start/saladbench_7b_base_noncoder/embeddings \
  --roles assistant \
  --model "Qwen/Qwen3-Embedding-4B"

echo ""
echo "=== Step 02: CP-2 Tensor Decomposition (CPU) ==="
python quick_start/code/02_run_cp2_from_generated_embeddings.py \
  --embedding-source saladbench_7b_base_noncoder \
  --max_rank 50

echo ""
echo "=== Step 03: Fit to Uncertainty ==="
python quick_start/code/03_fit_to_uncertainty_generated.py saladbench_7b_base_noncoder

echo ""
echo "=== Step 04: Grade Safety ==="
python quick_start/code/grade_saladbench.py quick_start/saladbench_7b_base_noncoder \
  --judge keyword

echo "--- EVALUATION: ANY UNSAFE (STRICT) ---"
python -m matu.evaluate_uncertainty \
  --uncertainty quick_start/saladbench_7b_base_noncoder/results/uncertainty_generated.pkl \
  --labels quick_start/saladbench_7b_base_noncoder/results/accuracy_dict_generated.pkl \
  --score_mode raw \
  --error_rule any_incorrect

echo ""
echo "--- EVALUATION: MAJORITY UNSAFE (RELAXED) ---"
python -m matu.evaluate_uncertainty \
  --uncertainty quick_start/saladbench_7b_base_noncoder/results/uncertainty_generated.pkl \
  --labels quick_start/saladbench_7b_base_noncoder/results/accuracy_dict_generated.pkl \
  --score_mode raw \
  --error_rule majority_incorrect

echo "SALADBENCH MATU PIPELINE DONE FOR 7b_base_noncoder!"
