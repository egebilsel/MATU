#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J sb-14b_base
#SBATCH --chdir=/dl_scratch2/ege/ege/repos/MATU
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:1
#SBATCH --output=/dl_scratch2/ege/ege/repos/MATU/results/saladbench_14b_base-%j.out
#SBATCH --error=/dl_scratch2/ege/ege/repos/MATU/results/saladbench_14b_base-%j.err
#SBATCH --time=3-00:00:00

# 1. Environment ve Cache Ayarları
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/ege/envs/matu-uq/

export HF_HOME=/dl_scratch2/ege/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /dl_scratch2/ege/ege/repos/MATU

# Çıktı klasörlerini oluştur
mkdir -p quick_start/saladbench_14b_base/embeddings
mkdir -p quick_start/saladbench_14b_base/results
mkdir -p quick_start/data/saladbench

echo "=== Step 00: Generate SaladBench Logs (GPU) ==="
python examples/generate_logs_saladbench.py \
  --max_items 100 \
  --model "Qwen/Qwen2.5-Coder-14B-Instruct" \
  --runs 10 \
  --seed 42 \
  --out quick_start/saladbench_14b_base/conversation_logs_saladbench.json

echo ""
echo "=== Step 01: Embed Logs (GPU) ==="
python -m matu.embed_logs \
  --logs quick_start/saladbench_14b_base/conversation_logs_saladbench.json \
  --out_dir quick_start/saladbench_14b_base/embeddings \
  --roles user assistant \
  --model "Qwen/Qwen3-Embedding-0.6B"

echo ""
echo "=== Step 02: CP-2 Tensor Decomposition (CPU) ==="
python quick_start/code/02_run_cp2_from_generated_embeddings.py \
  --embedding-source saladbench_14b_base \
  --max_rank 50

echo ""
echo "=== Step 03: Fit to Uncertainty ==="
python quick_start/code/03_fit_to_uncertainty_generated.py saladbench_14b_base

echo ""
echo "=== Step 04: Grade Safety ==="
python quick_start/code/grade_saladbench.py quick_start/saladbench_14b_base \
  --judge keyword

echo "--- EVALUATION: ANY UNSAFE (STRICT) ---"
python -m matu.evaluate_uncertainty \
  --uncertainty quick_start/saladbench_14b_base/results/uncertainty_generated.pkl \
  --labels quick_start/saladbench_14b_base/results/accuracy_dict_generated.pkl \
  --score_mode raw \
  --error_rule any_incorrect

echo ""
echo "--- EVALUATION: MAJORITY UNSAFE (RELAXED) ---"
python -m matu.evaluate_uncertainty \
  --uncertainty quick_start/saladbench_14b_base/results/uncertainty_generated.pkl \
  --labels quick_start/saladbench_14b_base/results/accuracy_dict_generated.pkl \
  --score_mode raw \
  --error_rule majority_incorrect

echo "SALADBENCH MATU PIPELINE DONE FOR 14b_base!"
