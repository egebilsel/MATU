#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J sb-3b_e_time_exp
#SBATCH --chdir=/dl_scratch2/ege/ege/repos/MATU
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:1
#SBATCH --output=/dl_scratch2/ege/ege/repos/MATU/results/saladbench_3b_evolved_time_exp-%j.out
#SBATCH --error=/dl_scratch2/ege/ege/repos/MATU/results/saladbench_3b_evolved_time_exp-%j.err
#SBATCH --time=2-00:00:00

# 1. Environment ve Cache Ayarları
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/ege/envs/matu-uq/

export HF_HOME=/dl_scratch2/ege/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /dl_scratch2/ege/ege/repos/MATU

# Çıktı klasörlerini oluştur
mkdir -p quick_start/saladbench_3b_evolved_time_exp/embeddings
mkdir -p quick_start/saladbench_3b_evolved_time_exp/results

# Sadece önceki log dosyasını kopyalıyoruz (Yeniden inference yapmamak için)
cp quick_start/saladbench_3b_evolved/conversation_logs_saladbench.json quick_start/saladbench_3b_evolved_time_exp/

echo ""
echo "=== Step 01: Embed Logs (GPU) ==="
python -m matu.embed_logs \
  --logs quick_start/saladbench_3b_evolved_time_exp/conversation_logs_saladbench.json \
  --out_dir quick_start/saladbench_3b_evolved_time_exp/embeddings \
  --roles user assistant \
  --model "Qwen/Qwen3-Embedding-0.6B"

echo ""
echo "=== Step 02: CP-2 Tensor Decomposition (CPU) ==="
python quick_start/code/02_run_cp2_from_generated_embeddings.py \
  --embedding-source saladbench_3b_evolved_time_exp \
  --max_rank 50 \
  --time-weighting exp

echo ""
echo "=== Step 03: Fit to Uncertainty ==="
python quick_start/code/03_fit_to_uncertainty_generated.py saladbench_3b_evolved_time_exp

echo ""
echo "=== Step 04: Grade Safety ==="
python quick_start/code/grade_saladbench.py quick_start/saladbench_3b_evolved_time_exp \
  --judge keyword

echo "--- EVALUATION: ANY UNSAFE (STRICT) ---"
python -m matu.evaluate_uncertainty \
  --uncertainty quick_start/saladbench_3b_evolved_time_exp/results/uncertainty_generated.pkl \
  --labels quick_start/saladbench_3b_evolved_time_exp/results/accuracy_dict_generated.pkl \
  --score_mode raw \
  --error_rule any_incorrect

echo ""
echo "--- EVALUATION: MAJORITY UNSAFE (RELAXED) ---"
python -m matu.evaluate_uncertainty \
  --uncertainty quick_start/saladbench_3b_evolved_time_exp/results/uncertainty_generated.pkl \
  --labels quick_start/saladbench_3b_evolved_time_exp/results/accuracy_dict_generated.pkl \
  --score_mode raw \
  --error_rule majority_incorrect

echo "SALADBENCH MATU PIPELINE DONE FOR 3b_evolved_time_exp!"
