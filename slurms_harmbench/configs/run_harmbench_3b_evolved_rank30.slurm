#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J hb-3b_e_rank30
#SBATCH --chdir=/dl_scratch2/ege/ege/repos/MATU
#SBATCH --nodelist=romer3
#SBATCH --gres=gpu:1
#SBATCH --output=/dl_scratch2/ege/ege/repos/MATU/results/harmbench_3b_evolved_rank30-%j.out
#SBATCH --error=/dl_scratch2/ege/ege/repos/MATU/results/harmbench_3b_evolved_rank30-%j.err
#SBATCH --time=2-00:00:00

# 1. Environment ve Cache Ayarları
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/ege/envs/matu-uq/

export HF_HOME=/dl_scratch2/ege/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /dl_scratch2/ege/ege/repos/MATU

# Çıktı klasörlerini oluştur
mkdir -p quick_start/harmbench_3b_evolved_rank30/embeddings
mkdir -p quick_start/harmbench_3b_evolved_rank30/results

# Sadece önceki log dosyasını kopyalıyoruz (Yeniden inference yapmamak için)
cp quick_start/harmbench_3b_evolved/conversation_logs_harmbench.json quick_start/harmbench_3b_evolved_rank30/

echo ""
echo "=== Step 01: Embed Logs (GPU) ==="
python -m matu.embed_logs \
  --logs quick_start/harmbench_3b_evolved_rank30/conversation_logs_harmbench.json \
  --out_dir quick_start/harmbench_3b_evolved_rank30/embeddings \
  --roles user assistant \
  --model "Qwen/Qwen3-Embedding-0.6B"

echo ""
echo "=== Step 02: CP-2 Tensor Decomposition (CPU) ==="
python quick_start/code/02_run_cp2_from_generated_embeddings.py \
  --embedding-source harmbench_3b_evolved_rank30 \
  --max_rank 30

echo ""
echo "=== Step 03: Fit to Uncertainty ==="
python quick_start/code/03_fit_to_uncertainty_generated.py harmbench_3b_evolved_rank30

echo ""
echo "=== Step 04: Grade Safety ==="
python quick_start/code/grade_harmbench.py quick_start/harmbench_3b_evolved_rank30 \
  --judge llm_judge

echo "--- EVALUATION: ANY UNSAFE (STRICT) ---"
python -m matu.evaluate_uncertainty \
  --uncertainty quick_start/harmbench_3b_evolved_rank30/results/uncertainty_generated.pkl \
  --labels quick_start/harmbench_3b_evolved_rank30/results/accuracy_dict_generated.pkl \
  --score_mode raw \
  --error_rule any_incorrect

echo ""
echo "--- EVALUATION: MAJORITY UNSAFE (RELAXED) ---"
python -m matu.evaluate_uncertainty \
  --uncertainty quick_start/harmbench_3b_evolved_rank30/results/uncertainty_generated.pkl \
  --labels quick_start/harmbench_3b_evolved_rank30/results/accuracy_dict_generated.pkl \
  --score_mode raw \
  --error_rule majority_incorrect

echo "HARMBENCH MATU PIPELINE DONE FOR 3b_evolved_rank30!"
