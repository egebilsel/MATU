#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J matu-generation
#SBATCH --chdir=/dl_scratch2/ege/ege/repos/MATU
#SBATCH --nodelist=adep1
#SBATCH --gres=gpu:1
#SBATCH --output=/dl_scratch2/ege/ege/repos/MATU/results/gen_matu-%j.out
#SBATCH --error=/dl_scratch2/ege/ege/repos/MATU/results/gen_matu-%j.err
#SBATCH --time=2-00:00:00

# 1. Environment ve Cache Ayarları
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/ege/envs/matu-uq/

export HF_HOME=/dl_scratch2/ege/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /dl_scratch2/ege/ege/repos/MATU

# Çıktı klasörlerini oluştur (Hata vermemesi için)
mkdir -p quick_start/generated/embeddings
mkdir -p quick_start/generated/results

echo "=== Step 00: Generate Logs (GPU) ==="
# Bu script belirlenen modeli indirip, MATH datasetinden max_items kadar soru çekecek
# ve her soru için runs kadar deneme (farklı cevap) üretecek.
python examples/generate_logs_hf_qwen.py \
  --model "Qwen/Qwen2.5-3B-Instruct" \
  --data_root quick_start/data/MATH/test \
  --max_items 5 \
  --runs 10 \
  --out quick_start/generated/conversation_logs_hf_qwen.json

echo ""
echo "=== Step 01: Embed Logs (GPU) ==="
# Üretilen logları okuyup Qwen3 modeliyle embedding matrislerine çevirir
python -m matu.embed_logs \
  --logs quick_start/generated/conversation_logs_hf_qwen.json \
  --out_dir quick_start/generated/embeddings \
  --roles user assistant \
  --model "Qwen/Qwen3-Embedding-0.6B"

echo ""
echo "=== Step 02: CP-2 Tensor Decomposition (CPU) ==="
# Üretilen ve anında embed edilen logları alıp tensor ayrıştırma işlemini yapacak
python quick_start/code/02_run_cp2_from_generated_embeddings.py \
  --embedding-source generated \
  --max_rank 50

echo ""
echo "=== Step 03: Fit to Uncertainty ==="
# CP-2'den çıkan sonuçları skaler belirsizlik değerlerine çevirecek
python quick_start/code/03_fit_to_uncertainty_generated.py

echo ""
echo "=== Step 04: Evaluate Generated ==="
# Ürettiğimiz verilerin AUROC / AUARC gibi başarı ve belirsizlik skorlarını yazdıracak
python quick_start/code/04_evaluate_generated_results.py

echo "MATU GENERATION AND PIPELINE DONE!"
