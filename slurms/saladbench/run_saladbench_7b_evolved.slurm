#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J sb-7b_evol
#SBATCH --chdir=/dl_scratch2/ege/ege/repos/MATU
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:1
#SBATCH --output=/dl_scratch2/ege/ege/repos/MATU/results/saladbench/saladbench_7b_evolved-%j.out
#SBATCH --error=/dl_scratch2/ege/ege/repos/MATU/results/saladbench/saladbench_7b_evolved-%j.err
#SBATCH --time=3-00:00:00

# 1. Environment and Cache Settings
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/ege/envs/matu-uq/

export HF_HOME=/dl_scratch2/ege/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /dl_scratch2/ege/ege/repos/MATU

# Create output directories
mkdir -p quick_start/saladbench/saladbench_7b_evolved/embeddings
mkdir -p quick_start/saladbench/saladbench_7b_evolved/results
mkdir -p quick_start/data/saladbench

echo "=== Step 00: Generate SaladBench Logs (GPU) ==="
python examples/generate_logs_saladbench.py \
  --max_items 100 \
  --model "andrewzh/Absolute_Zero_Reasoner-Coder-7b" \
  --runs 10 \
  --seed 42 \
  --out quick_start/saladbench/saladbench_7b_evolved/conversation_logs_saladbench.json

echo ""
echo "=== Step 01: Embed Logs (GPU) ==="
python -m matu.embed_logs \
  --logs quick_start/saladbench/saladbench_7b_evolved/conversation_logs_saladbench.json \
  --out_dir quick_start/saladbench/saladbench_7b_evolved/embeddings \
  --roles user assistant \
  --model "Qwen/Qwen3-Embedding-0.6B"

echo ""
echo "=== Step 02: CP-2 Tensor Decomposition (CPU) ==="
python quick_start/code/02_run_cp2_from_generated_embeddings.py \
  --embedding-source saladbench/saladbench_7b_evolved \
  --max_rank 50

echo ""
echo "=== Step 03: Fit to Uncertainty ==="
python quick_start/code/03_fit_to_uncertainty_generated.py saladbench_7b_evolved

echo ""
echo "=== Step 04: Grade Safety ==="
python quick_start/code/grade_saladbench.py quick_start/saladbench/saladbench_7b_evolved \
  --judge llm_judge \
  --judge_model "Qwen/Qwen2.5-14B-Instruct"

echo "--- EVALUATION: ANY UNSAFE (STRICT) ---"
python -m matu.evaluate_uncertainty \
  --uncertainty quick_start/saladbench/saladbench_7b_evolved/results/uncertainty_generated.pkl \
  --labels quick_start/saladbench/saladbench_7b_evolved/results/accuracy_dict_generated.pkl \
  --score_mode raw \
  --error_rule any_incorrect

echo ""
echo "--- EVALUATION: MAJORITY UNSAFE (RELAXED) ---"
python -m matu.evaluate_uncertainty \
  --uncertainty quick_start/saladbench/saladbench_7b_evolved/results/uncertainty_generated.pkl \
  --labels quick_start/saladbench/saladbench_7b_evolved/results/accuracy_dict_generated.pkl \
  --score_mode raw \
  --error_rule majority_incorrect

echo "SALADBENCH MATU PIPELINE DONE FOR 7b_evolved!"
