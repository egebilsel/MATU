# Quick Start Code

These scripts make the quick start reproducible step by step.

They read local settings from `quick_start/.env`. Create it from
`quick_start/.env.example` before running generation scripts.

Run from the repository root:

```bash
python quick_start/code/04_evaluate_reference_results.py
```

This skips all earlier steps and evaluates the included reference MATU result.
The same evaluator can also reproduce the paper-matching MMLU + AutoGen +
Qwen2.5 result:

```bash
python quick_start/code/04_evaluate_reference_results.py --sample mmlu-autogen-qwen
python quick_start/code/04_evaluate_reference_results.py --sample all
```

To inspect or reuse the included embedding archives:

```bash
mkdir -p quick_start/generated/reference_embeddings/math
python -m zipfile -e quick_start/data/embeddings_Math_qwen2.5_qwen3.zip quick_start/generated/reference_embeddings/math

mkdir -p quick_start/generated/reference_embeddings/mmlu_autogen_qwen
python -m zipfile -e quick_start/data/embeddings_MMLU_Autogen_qwen2.5.zip quick_start/generated/reference_embeddings/mmlu_autogen_qwen
```

To regenerate intermediate outputs:

```bash
python quick_start/code/01_embed_reference_logs.py
python quick_start/code/02_run_cp2_from_generated_embeddings.py
python quick_start/code/03_fit_to_uncertainty_generated.py
python quick_start/code/04_evaluate_generated_results.py
```

The included reference embeddings in `quick_start/data/` are Qwen3 embeddings
with shape `(num_steps, 1024)`, stored as zip archives rather than raw pickle
files.

Optional log generation examples:

```bash
python quick_start/code/00_generate_logs_camel_gpt.py
python quick_start/code/00_generate_logs_hf_qwen.py
```
