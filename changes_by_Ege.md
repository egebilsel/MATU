# Code Changes
analysis/analyze_harmbench_uncertainty.py:
- created by Ege for HarmBench
- added to calculate AUROC, AUARC, safe rate, and base vs evolved comparisons for HarmBench experiments

analysis/analyze_saladbench_uncertainty.py:
- created by Ege for SaladBench experiments
- added to calculate AUROC, AUARC, safe rate, and base vs evolved comparisons for SaladBench experiments

analysis/analyze_math_uncertainty.py:
- created by Ege for MATH experiments
- added to calculate AUROC, AUARC, safe rate, and base vs evolved comparisons for MATH experiments

baselines/run_eigv_analysis.py:
- created by Ege 
- added to calculate Eigenvalue (EigV) / NLI uncertainty metric for MATH experiments to compare the MATU and EigV

examples/generate_logs_harmbench.py:
- created by Ege for harmbench experiments
- used for generating logs for harmbench

examples/generate_logs_saladbench.py:
- created by Ege for saladbench experiments
- used for generating logs for saladbench

matu/cp2_matu.py:
- changed by Ege for time weighting configs
- added time weighting options to the script
- comments are added to the changes in cp2_matu.py

matu/remove_stopwords_llm.py:
- created by Ege for cleaned config
- used for removing stopwords from the logs

quick_start/code/02_run_cp2_from_generated_embeddings.py:
- changed by Ege 
- argparse added for configs
- input output paths are adjusted for ease of use

quick_start/code/03_fit_to_uncertainty_generated.py:
- changed by Ege 
- added a command-line argument for target directory to support custom run folders (e.g., for slurm jobs)

quick_start/code/03_fit_to_uncertainty_reference.py:
- changed by Ege 
- reverted baseline paths from 'Assistonly' specific naming back to general default naming
- Note: This change is mostly unimportant as the script is only used for the pre-packaged reference data.

quick_start/code/04_evaluate_reference_results.py:
- changed by Ege 
- reverted baseline paths from 'Assistonly' specific naming back to general default naming
- Note: This change is mostly unimportant as the script is only used for evaluating the pre-packaged reference data.

quick_start/code/05_evaluate_baselines.py:
- changed by Ege 
- added argparse support to select which baseline sample to evaluate (math-qwen, mmlu-autogen-qwen, or all)
- Note: This script is entirely unused in the main generation/experiment pipeline. It is only included for evaluating pre-packaged baseline (SAUP) results, making these changes unimportant for actual experiments.

quick_start/code/grade_math.py:
- created/changed by Ege
- added a dedicated script to grade MATH dataset responses automatically

quick_start/code/grade_harmbench.py:
- created by Ege
- added to evaluate HarmBench responses
- supports both fast keyword-based matching and LLM Judge evaluation
- recently updated to default to the LLM judge with an improved prompt

quick_start/code/grade_saladbench.py:
- created by Ege
- added to evaluate SaladBench responses
- supports both fast keyword-based matching and LLM Judge evaluation
- recently updated to default to the LLM judge and fixed hardcoded paths


# Directories
quick_start/data/
- changed by Ege
- SaladBench and HarmBench is downloaded under this directory

quick_start/math/
- created by Ege
- it includes the logs of the math experiments

quick_start/harmbench/
- created by Ege
- it includes the logs of the harmbench experiments

quick_start/saladbench/
- created by Ege
- it includes the logs of the saladbench experiments

quick_start/reference_results/
- created by Ege
- it includes the results of the reference experiments

results/
- created by Ege
- it includes the results of the experiments

scatter_plots/
- created by Ege
- it includes the scatter plots of the experiments
