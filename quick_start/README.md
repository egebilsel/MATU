# Quick Start

This folder contains ready-to-run MATU sample artifacts. You can get the main
results immediately, or optionally regenerate intermediate files step by step.

The important point: **the logs, labels, fit dictionaries, scalar uncertainty,
baseline scores, and zipped reference embeddings are already provided.** You do
not need to re-embed logs or rerun CP-2 unless you want to inspect the pipeline
internals.

## Environment

Quick-start scripts read local settings from:

```text
quick_start/.env
```

Create it from the template:

```bash
cp quick_start/.env.example quick_start/.env
```

On Windows PowerShell:

```powershell
Copy-Item quick_start/.env.example quick_start/.env
```

Important variables:

```text
OPENAI_API_KEY=                 # only needed for CAMEL/GPT log generation
MATU_OPENAI_MODEL=GPT_4O
MATU_QWEN_MODEL=Qwen/Qwen2.5-7B-Instruct
MATU_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
MATU_MODEL_CACHE=               # optional local HF cache/model path
MATU_MAX_RANK=50
```

The included reference evaluation does not require `OPENAI_API_KEY`.

## Included Files

| File | Description |
| --- | --- |
| `data/conversation_logs_Math_qwen2.5.json` | Ready-to-use MATH repeated conversation logs. |
| `data/conversation_logs_MMLU_Autogen_qwen2.5.json` | MMLU AutoGen conversation log sample for the paper-matching result. |
| `data/embeddings_Math_qwen2.5_qwen3.zip` | Zipped Qwen3 user and assistant embedding matrices for MATH. |
| `data/embeddings_MMLU_Autogen_qwen2.5.zip` | Zipped AutoGen analyst, verifier, and star embedding matrices for MMLU. |
| `data/embedding_metadata.json` | Embedding model metadata. |
| `results/fit_dict_Math_Assistonly_qwen2.5_qwen3embedding.pkl` | Ready-to-use MATU fit curves for MATH. |
| `results/uncertainty_Math_Assistonly_qwen2.5.pkl` | Ready-to-use scalar MATU uncertainty for MATH. |
| `results/accuracy_dict_Math_qwen2.5.pkl` | MATH repeated-run correctness labels. |
| `results/fit_dict_MMLU_Autogen_qwen2.5.pkl` | Ready-to-use MATU fit curves for MMLU AutoGen. |
| `results/accuracy_dict_MMLU_Autogen_qwen2.5.pkl` | MMLU AutoGen repeated-run correctness labels. |
| `results/saup_scores_Math_qwen2.5.pkl` | Ready-to-use SAUP-Multiple baseline scores. |

## Fastest Result Check

Run from the repository root:

```bash
python quick_start/code/04_evaluate_reference_results.py --sample all
```

This script loads the included result files and prints metrics. It does not
download models, call APIs, regenerate embeddings, or rerun CP-2.

Expected output:

```text
MATH + Qwen2.5-7B
Tasks: 400
Mean accuracy: 0.8383
AUROC: 0.7205
AUARC: 0.9017

Paper Table 2: MMLU + AutoGen + Qwen2.5-7B
Tasks: 400
Mean accuracy: 0.7785
AUROC: 0.7315
AUARC: 0.8834
```

The MATH result was re-run for public release packaging, so it is close to but
not bit-for-bit identical to the paper table. The MMLU AutoGen result uses the
original paper artifact and matches Table 2 up to display rounding.

## Quick-Start Workflow

| Step | Command | What It Does | Required? | Output |
| --- | --- | --- | --- | --- |
| Read included MATU results | `python quick_start/code/04_evaluate_reference_results.py --sample all` | Loads provided MATU fit/uncertainty files and labels, then prints MATH and MMLU metrics. | Yes, for fastest verification. | Console AUROC/AUARC. |
| Read included baseline | `python quick_start/code/05_evaluate_baselines.py` | Loads the provided SAUP-Multiple score file and evaluates it against MATH labels. | Optional. | Baseline AUROC/AUARC. |
| Extract MATH reference embeddings | `python -m zipfile -e quick_start/data/embeddings_Math_qwen2.5_qwen3.zip quick_start/generated/reference_embeddings/math` | Unzips the included MATH raw embedding matrices. | Optional; only needed if you want to inspect embeddings or rerun CP-2 from them. | Two raw embedding `.pkl` files. |
| Extract MMLU reference embeddings | `python -m zipfile -e quick_start/data/embeddings_MMLU_Autogen_qwen2.5.zip quick_start/generated/reference_embeddings/mmlu_autogen_qwen` | Unzips the included MMLU AutoGen raw embedding matrices. | Optional; only for inspection/reuse. | Three raw embedding `.pkl` files. |
| Re-embed MATH logs | `python quick_start/code/01_embed_reference_logs.py` | Recomputes Qwen3 embeddings from the included MATH conversation log. | Optional; zipped reference embeddings are already included. | `quick_start/generated/embeddings/*.pkl`. |
| Run CP-2 from generated embeddings | `python quick_start/code/02_run_cp2_from_generated_embeddings.py` | Runs MATU/CP-2 on embeddings created by the previous step. | Optional; public fit curves are already in `results/`. | `quick_start/generated/results/matu_scores.pkl` and `fit_dict_generated.pkl`. |
| Convert generated fit curves | `python quick_start/code/03_fit_to_uncertainty_generated.py` | Converts generated fit curves to scalar MATU uncertainty. | Optional; only after generated CP-2. | `quick_start/generated/results/uncertainty_generated.pkl`. |
| Evaluate generated run | `python quick_start/code/04_evaluate_generated_results.py` | Evaluates regenerated uncertainty against included MATH labels. | Optional; verifies a regenerated pipeline. | Console AUROC/AUARC. |

## Unzip Reference Embeddings

Create output folders first:

```bash
mkdir -p quick_start/generated/reference_embeddings/math
mkdir -p quick_start/generated/reference_embeddings/mmlu_autogen_qwen
```

Extract the MATH embeddings:

```bash
python -m zipfile -e quick_start/data/embeddings_Math_qwen2.5_qwen3.zip quick_start/generated/reference_embeddings/math
```

This creates:

```text
quick_start/generated/reference_embeddings/math/user_embedding_matrices_Math_qwen2.5_qwen3.pkl
quick_start/generated/reference_embeddings/math/assistant_embedding_matrices_Math_qwen2.5_qwen3.pkl
```

Extract the MMLU AutoGen embeddings:

```bash
python -m zipfile -e quick_start/data/embeddings_MMLU_Autogen_qwen2.5.zip quick_start/generated/reference_embeddings/mmlu_autogen_qwen
```

This creates:

```text
quick_start/generated/reference_embeddings/mmlu_autogen_qwen/autogen_analyst_embedding_matrices_MMLU_HF_qwen2.5.pkl
quick_start/generated/reference_embeddings/mmlu_autogen_qwen/autogen_verifier_embedding_matrices_MMLU_HF_qwen2.5.pkl
quick_start/generated/reference_embeddings/mmlu_autogen_qwen/autogen_star_embedding_matrices_MMLU_HF_qwen2.5.pkl
```

## Optional: Rerun CP-2 From Included Embeddings

Use this path when you want to verify MATU/CP-2 from raw embeddings but do not
want to download the embedding model. The public `fit_dict` is already provided,
so this is not required for the fastest result check.

```bash
mkdir -p quick_start/generated/results
python -m matu.cp2_matu \
  --embeddings \
  quick_start/generated/reference_embeddings/math/user_embedding_matrices_Math_qwen2.5_qwen3.pkl \
  quick_start/generated/reference_embeddings/math/assistant_embedding_matrices_Math_qwen2.5_qwen3.pkl \
  --out quick_start/generated/results/matu_scores.pkl \
  --legacy_fit_out quick_start/generated/results/fit_dict_generated.pkl \
  --max_rank 50
```

After CP-2 finishes, convert the generated fit curves into scalar uncertainty.
This script reads `quick_start/generated/results/fit_dict_generated.pkl` and
writes `quick_start/generated/results/uncertainty_generated.pkl`.

```bash
python quick_start/code/03_fit_to_uncertainty_generated.py
```

Finally, evaluate the regenerated uncertainty against the included MATH labels.
This checks the output of the optional regenerated pipeline, not the already
provided reference result.

```bash
python quick_start/code/04_evaluate_generated_results.py
```

## Optional: Re-Embed Logs From Scratch

Use this path only if you want to verify the embedding stage itself. It loads or
downloads `Qwen/Qwen3-Embedding-0.6B`; a GPU is faster, but the provided
artifacts let users skip this step.

```bash
python quick_start/code/01_embed_reference_logs.py
```

Output:

```text
quick_start/generated/embeddings/user_embedding_matrices.pkl
quick_start/generated/embeddings/assistant_embedding_matrices.pkl
```

After re-embedding, run CP-2 on the newly generated embeddings. This wrapper
expects the files produced by `01_embed_reference_logs.py` and writes generated
MATU scores under `quick_start/generated/results/`.

```bash
python quick_start/code/02_run_cp2_from_generated_embeddings.py
```

Then convert the generated fit curves into scalar uncertainty:

```bash
python quick_start/code/03_fit_to_uncertainty_generated.py
```

Finally, evaluate the regenerated uncertainty against the included MATH labels:

```bash
python quick_start/code/04_evaluate_generated_results.py
```

## Optional: Generate New Logs

The included logs are already available, so generation is not needed for the
quick-start result. These scripts are examples for users who want to collect new
conversation logs.

CAMEL/OpenAI example:

```bash
python quick_start/code/00_generate_logs_camel_gpt.py
```

HF/Qwen example:

```bash
python quick_start/code/00_generate_logs_hf_qwen.py
```

These write new files under:

```text
quick_start/generated/
quick_start/generated/embeddings_from_camel_gpt_generation/
quick_start/generated/embeddings_from_hf_qwen_generation/
```

## Baselines

Evaluate the included SAUP-Multiple reference result:

```bash
python quick_start/code/05_evaluate_baselines.py
```

Expected output:

```text
SAUP-Multiple:
  tasks: 400
  AUROC: 0.6097
  AUARC: 0.8722
```

EigV code is provided in the root baseline script. Its result file is not
included in this quick start:

```bash
python baselines/eigv.py \
  --logs quick_start/data/conversation_logs_Math_qwen2.5.json \
  --mode final \
  --out quick_start/generated/results/eigv_final.pkl
```
