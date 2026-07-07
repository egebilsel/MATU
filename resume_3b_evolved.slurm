#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J resume-3b_evolved
#SBATCH --chdir=/dl_scratch2/ege/ege/repos/MATU
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:1
#SBATCH --output=/dl_scratch2/ege/ege/repos/MATU/results/resume_3b_evolved-%j.out
#SBATCH --error=/dl_scratch2/ege/ege/repos/MATU/results/resume_3b_evolved-%j.err
#SBATCH --time=02:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/ege/envs/matu-uq/
export PYTHONPATH=$(pwd)

echo "=== Step 02: CP-2 Tensor Decomposition (CPU) ==="
python quick_start/code/02_run_cp2_from_generated_embeddings.py \
  --embedding-source gen_3b_evolved \
  --max_rank 50

echo ""
echo "=== Step 03: Fit to Uncertainty ==="
python quick_start/code/03_fit_to_uncertainty_generated.py gen_3b_evolved

echo ""
echo "=== Step 04: Evaluate Generated ==="
python quick_start/code/grade_math.py quick_start/gen_3b_evolved

python -m matu.evaluate_uncertainty \
  --uncertainty quick_start/gen_3b_evolved/results/uncertainty_generated.pkl \
  --labels quick_start/gen_3b_evolved/results/accuracy_dict_generated.pkl \
  --score_mode raw

echo "MATU PIPELINE RESUMED AND DONE FOR 3b_evolved!"
