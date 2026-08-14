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

analysis/analyze_harmbench_uncertainty.py:
- created by Ege
- added to calculate AUROC, AUARC, safe rate, and base vs evolved comparisons for HarmBench experiments

analysis/analyze_saladbench_uncertainty.py:
- created by Ege
- added to calculate AUROC, AUARC, safe rate, and base vs evolved comparisons for SaladBench experiments

analysis/analyze_math_uncertainty.py:
- created by Ege
- added to calculate AUROC, AUARC, safe rate, and base vs evolved comparisons for MATH experiments

baselines/run_eigv_analysis.py:
- created by Ege 
- added to calculate Eigenvalue (EigV) / NLI uncertainty metric for MATH experiments


## 14 August 2026
- `quick_start/` klasörü altındaki deney çıktı klasörleri 3 ana klasör altına ayrıldı (`math`, `harmbench`, `saladbench`). Ortak veriler (`data/`, `results/`) olduğu gibi bırakıldı.
- Generator scriptleri (`generate_harmbench_slurms.py`, `generate_saladbench_slurms.py`, `generate_time_weighting_slurms.py`) silinmişti, git üzerinden kurtarılıp output pathleri düzenlendi ve slurm dosyaları yeni pathlere göre yeniden üretildi.
- Bütün `slurms/` altındaki `.slurm` dosyalarının içeriğindeki `quick_start/` pathleri otomatik bir script aracılığıyla ilgili deney klasörlerini işaret edecek şekilde (`math/`, `harmbench/`, `saladbench/`) güncellendi.
- Analiz scriptleri (`analysis/analyze_*.py`), değerlendirme (grader) scriptleri (`grade_*.py`) ve plot scriptleri yeni dizin yapısını destekleyecek şekilde güncellendi.
- `README.md` içerisindeki eski `quick_start/generated` örnekleri, yeni `quick_start/math/generated` klasörünü gösterecek şekilde güncellendi.
