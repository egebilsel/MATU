#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J regrade-all
#SBATCH --chdir=/dl_scratch2/ege/ege/repos/MATU
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:1
#SBATCH --output=/dl_scratch2/ege/ege/repos/MATU/results/harmbench/regrade_harmbench-%j.out
#SBATCH --error=/dl_scratch2/ege/ege/repos/MATU/results/harmbench/regrade_harmbench-%j.err
#SBATCH --time=1-00:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/ege/envs/matu-uq/

export HF_HOME=/dl_scratch2/ege/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /dl_scratch2/ege/ege/repos/MATU

# 1. Main (Base/Evolved) Models - LLM Judge will run for these
MAIN_EXPERIMENTS=(
  "harmbench_3b_base"
  "harmbench_3b_evolved"
  "harmbench_7b_base"
  "harmbench_7b_evolved"
  "harmbench_14b_base"
  "harmbench_14b_evolved"
  "harmbench_7b_base_noncoder"
  "harmbench_7b_evolved_noncoder"
)

# 2. Derivative Configs - These will directly copy the grading results of the Main model
CONFIG_EXPERIMENTS=(
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

# A. GRADE MAIN EXPERIMENTS
for EXP in "${MAIN_EXPERIMENTS[@]}"; do
  DIR="quick_start/${EXP}"
  
  if [ ! -f "${DIR}/conversation_logs_harmbench.json" ]; then
    echo "SKIP MAIN: ${EXP} (log file not found)"
    continue
  fi

  echo ""
  echo "================================================================"
  echo "RE-GRADING MAIN: ${EXP}"
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

  echo "--- EVALUATION: MAJORITY UNSAFE (RELAXED) ---"
  python -m matu.evaluate_uncertainty \
    --uncertainty "${DIR}/results/uncertainty_generated.pkl" \
    --labels "${DIR}/results/accuracy_dict_generated.pkl" \
    --score_mode raw \
    --error_rule majority_incorrect
done

# B. FAST EVALUATE DERIVATIVE EXPERIMENTS
for EXP in "${CONFIG_EXPERIMENTS[@]}"; do
  DIR="quick_start/${EXP}"
  
  if [ ! -f "${DIR}/results/uncertainty_generated.pkl" ]; then
    echo "SKIP CONFIG: ${EXP} (uncertainty not found)"
    continue
  fi

  echo ""
  echo "================================================================"
  echo "FAST EVALUATING CONFIG: ${EXP}"
  echo "================================================================"
  
  # Find the main model name (e.g. harmbench_3b_base_time_exp -> harmbench_3b_base)
  if [[ "$EXP" == *"_base_"* ]]; then
    SOURCE_DIR="quick_start/harmbench/harmbench_3b_base"
  else
    SOURCE_DIR="quick_start/harmbench/harmbench_3b_evolved"
  fi

  # Just copy the accuracy (grades) table
  cp "${SOURCE_DIR}/results/accuracy_dict_generated.pkl" "${DIR}/results/"

  echo "--- EVALUATION: ANY UNSAFE (STRICT) ---"
  python -m matu.evaluate_uncertainty \
    --uncertainty "${DIR}/results/uncertainty_generated.pkl" \
    --labels "${DIR}/results/accuracy_dict_generated.pkl" \
    --score_mode raw \
    --error_rule any_incorrect

  echo "--- EVALUATION: MAJORITY UNSAFE (RELAXED) ---"
  python -m matu.evaluate_uncertainty \
    --uncertainty "${DIR}/results/uncertainty_generated.pkl" \
    --labels "${DIR}/results/accuracy_dict_generated.pkl" \
    --score_mode raw \
    --error_rule majority_incorrect
done

echo ""
echo "================================================================"
echo "ALL RE-GRADING COMPLETE! Running analysis..."
echo "================================================================"
python analysis/analyze_harmbench_uncertainty.py

echo "RE-GRADING PIPELINE FINISHED!"
