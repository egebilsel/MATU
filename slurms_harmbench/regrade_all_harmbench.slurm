#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J regrade-all
#SBATCH --chdir=/dl_scratch2/ege/ege/repos/MATU
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:1
#SBATCH --output=/dl_scratch2/ege/ege/repos/MATU/results/regrade_harmbench-%j.out
#SBATCH --error=/dl_scratch2/ege/ege/repos/MATU/results/regrade_harmbench-%j.err
#SBATCH --time=1-00:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/ege/envs/matu-uq/

export HF_HOME=/dl_scratch2/ege/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /dl_scratch2/ege/ege/repos/MATU

# Tüm HarmBench deneylerini sadece LLM judge ile yeniden notlandır
EXPERIMENTS=(
  "harmbench_3b_base"
  "harmbench_3b_evolved"
  "harmbench_7b_base"
  "harmbench_7b_evolved"
  "harmbench_14b_base"
  "harmbench_14b_evolved"
  "harmbench_7b_base_noncoder"
  "harmbench_7b_evolved_noncoder"
  "harmbench_3b_base_cleaned"
  "harmbench_3b_evolved_cleaned"
  "harmbench_3b_base_embedding"
  "harmbench_3b_evolved_embedding"
  "harmbench_3b_base_rank30"
  "harmbench_3b_evolved_rank30"
  "harmbench_3b_base_assistant_only"
  "harmbench_3b_evolved_assistant_only"
  "harmbench_3b_base_user_only"
  "harmbench_3b_evolved_user_only"
  "harmbench_3b_base_time_exp"
  "harmbench_3b_evolved_time_exp"
  "harmbench_3b_base_time_linear"
  "harmbench_3b_evolved_time_linear"
  "harmbench_3b_base_time_cutoff"
  "harmbench_3b_evolved_time_cutoff"
  "harmbench_3b_base_time_sigmoid"
  "harmbench_3b_evolved_time_sigmoid"
  "harmbench_3b_base_time_log"
  "harmbench_3b_evolved_time_log"
  "harmbench_3b_base_assistant_embedding"
  "harmbench_3b_evolved_assistant_embedding"
)

for EXP in "${EXPERIMENTS[@]}"; do
  DIR="quick_start/${EXP}"
  
  # Loglar var mı kontrol et
  LOG_FILE="${DIR}/conversation_logs_harmbench.json"
  if [ ! -f "$LOG_FILE" ]; then
    # Config deneylerinde log farklı isimde olabilir
    LOG_FILE="${DIR}/cleaned_conversation_logs.json"
    if [ ! -f "$LOG_FILE" ]; then
      echo "SKIP: ${EXP} (log file not found)"
      continue
    fi
  fi

  echo ""
  echo "================================================================"
  echo "RE-GRADING: ${EXP}"
  echo "================================================================"
  
  python quick_start/code/grade_harmbench.py "${DIR}" \
    --judge llm_judge \
    --judge_model "Qwen/Qwen2.5-14B-Instruct"

  echo "--- EVALUATION: ANY UNSAFE (STRICT) ---"
  python -m matu.evaluate_uncertainty \
    --uncertainty "${DIR}/results/uncertainty_generated.pkl" \
    --labels "${DIR}/results/accuracy_dict_generated.pkl" \
    --score_mode raw \
    --error_rule any_incorrect

  echo ""
  echo "--- EVALUATION: MAJORITY UNSAFE (RELAXED) ---"
  python -m matu.evaluate_uncertainty \
    --uncertainty "${DIR}/results/uncertainty_generated.pkl" \
    --labels "${DIR}/results/accuracy_dict_generated.pkl" \
    --score_mode raw \
    --error_rule majority_incorrect

  echo "DONE: ${EXP}"
done

echo ""
echo "================================================================"
echo "ALL RE-GRADING COMPLETE! Running analysis..."
echo "================================================================"
python analyze_harmbench_uncertainty.py

echo "RE-GRADING PIPELINE FINISHED!"
