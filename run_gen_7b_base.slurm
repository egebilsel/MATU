#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J matu-7b_base
#SBATCH --chdir=/dl_scratch2/ege/ege/repos/MATU
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:1
#SBATCH --output=/dl_scratch2/ege/ege/repos/MATU/results/gen_7b_base-%j.out
#SBATCH --error=/dl_scratch2/ege/ege/repos/MATU/results/gen_7b_base-%j.err
#SBATCH --time=2-00:00:00

# 1. Environment ve Cache Ayarları
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/ege/envs/matu-uq/

export HF_HOME=/dl_scratch2/ege/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /dl_scratch2/ege/ege/repos/MATU

# Çıktı klasörlerini oluştur (Hata vermemesi için)
mkdir -p quick_start/gen_7b_base/embeddings
mkdir -p quick_start/gen_7b_base/results

echo "=== Step 00: Generate Logs (GPU) ==="
python examples/generate_logs_hf_qwen.py \
  --model "Qwen/Qwen2.5-Coder-7B-Instruct" \
  --data_root quick_start/data/MATH/test \
  --max_items 50 \
  --runs 10 \
  --seed 42 \
  --out quick_start/gen_7b_base/conversation_logs_hf_qwen.json

echo ""
echo "=== Step 01: Embed Logs (GPU) ==="
python -m matu.embed_logs \
  --logs quick_start/gen_7b_base/conversation_logs_hf_qwen.json \
  --out_dir quick_start/gen_7b_base/embeddings \
  --roles user assistant \
  --model "Qwen/Qwen3-Embedding-0.6B"

echo ""
echo "=== Step 02: CP-2 Tensor Decomposition (CPU) ==="
python quick_start/code/02_run_cp2_from_generated_embeddings.py \
  --embedding-source gen_7b_base \
  --max_rank 50

echo ""
echo "=== Step 03: Fit to Uncertainty ==="
python quick_start/code/03_fit_to_uncertainty_generated.py gen_7b_base

echo ""
echo "=== Step 04: Evaluate Generated ==="
python quick_start/code/grade_math.py quick_start/gen_7b_base

python -m matu.evaluate_uncertainty \
  --uncertainty quick_start/gen_7b_base/results/uncertainty_generated.pkl \
  --labels quick_start/gen_7b_base/results/accuracy_dict_generated.pkl \
  --score_mode raw

echo "MATU GENERATION AND PIPELINE DONE FOR 7b_base!"
