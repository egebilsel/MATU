# Quick Start

This folder contains one complete quick-start case copied from the original
experiments: MATH runs with Qwen2.5. It includes both code and reference outputs
for each stage.

You can either:

1. Skip all computation and directly evaluate the included final MATU result.
2. Re-run embedding and CP-2 from the included logs.
3. Optionally generate new logs with CAMEL/OpenAI or HF/Qwen examples.

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

Then edit `quick_start/.env`.

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

## Folder Layout

```text
code/
  00_generate_logs_camel_gpt.py
  00_generate_logs_hf_qwen.py
  01_embed_reference_logs.py
  02_run_cp2_from_generated_embeddings.py
  03_fit_to_uncertainty_reference.py
  03_fit_to_uncertainty_generated.py
  04_evaluate_reference_results.py
  04_evaluate_generated_results.py
  05_evaluate_baselines.py

data/
  conversation_logs_Math_qwen2.5.json
  conversation_logs_MMLU_Autogen_qwen2.5.json
  embeddings_Math_qwen2.5_qwen3.zip
  embeddings_MMLU_Autogen_qwen2.5.zip
  embedding_metadata.json

results/
  accuracy_dict_Math_qwen2.5.pkl
  fit_dict_Math_Assistonly_qwen2.5_qwen3embedding.pkl
  uncertainty_Math_Assistonly_qwen2.5.pkl
  accuracy_dict_MMLU_Autogen_qwen2.5.pkl
  fit_dict_MMLU_Autogen_qwen2.5.pkl
  saup_scores_Math_qwen2.5.pkl
```

## Fastest Path: Directly See Final Results

Run from the repository root:

```bash
python quick_start/code/04_evaluate_reference_results.py
```

This uses the final scalar uncertainty file:

```text
quick_start/results/uncertainty_Math_Assistonly_qwen2.5.pkl
```

If that file is missing, the script first converts the included `fit_dict`:

```text
quick_start/results/fit_dict_Math_Assistonly_qwen2.5_qwen3embedding.pkl
quick_start/results/accuracy_dict_Math_qwen2.5.pkl
```

and prints AUROC/AUARC directly.

Expected output:

```text
MATH + Qwen2.5-7B
Tasks: 400
Mean accuracy: 0.8383
AUROC: 0.7205
AUARC: 0.9017
```

## Paper-Matching Result

The quick start also includes a compact result that exactly matches the MATU
row for MMLU + AutoGen + Qwen2.5-7B in Table 2 of the paper:

```bash
python quick_start/code/04_evaluate_reference_results.py --sample mmlu-autogen-qwen
```

Expected output:

```text
Paper Table 2: MMLU + AutoGen + Qwen2.5-7B
Tasks: 400
Mean accuracy: 0.7785
AUROC: 0.7315
AUARC: 0.8834
```

This MMLU sample intentionally uses the legacy run-expanded AUROC protocol from
the paper tables. The general `matu eval` command reports a task-level error
AUROC, which is better for new code but does not reproduce the table value
exactly. To evaluate both included MATU samples in one command, run:

```bash
python quick_start/code/04_evaluate_reference_results.py --sample all
```

## Step 0: Optional Log Generation

The included reference logs are already available at:

```text
quick_start/data/conversation_logs_Math_qwen2.5.json
```

If users want to generate new logs, they can run either example.
Following the original experiment code, these generation examples also write
embedding matrices automatically when `--embedding_out_dir` is provided. The
quick-start wrappers already pass that argument.

CAMEL/OpenAI example:

```bash
python quick_start/code/00_generate_logs_camel_gpt.py
```

HF/Qwen example:

```bash
python quick_start/code/00_generate_logs_hf_qwen.py
```

These write new logs to:

```text
quick_start/generated/
quick_start/generated/embeddings_from_camel_gpt_generation/
quick_start/generated/embeddings_from_hf_qwen_generation/
```

They are examples only. The MATU framework can start from any collected logs
that follow `../data/LOG_FORMAT.md`.

## Step 1A: Use Included Reference Embeddings

The included embedding matrices are stored as zip archives so the public tree
does not contain large raw pickle files. To extract the MATH reference
embeddings:

```bash
mkdir -p quick_start/generated/reference_embeddings/math
python -m zipfile -e quick_start/data/embeddings_Math_qwen2.5_qwen3.zip quick_start/generated/reference_embeddings/math
```

This creates:

```text
quick_start/generated/reference_embeddings/math/user_embedding_matrices_Math_qwen2.5_qwen3.pkl
quick_start/generated/reference_embeddings/math/assistant_embedding_matrices_Math_qwen2.5_qwen3.pkl
```

To extract the MMLU + AutoGen + Qwen2.5 paper artifact embeddings:

```bash
mkdir -p quick_start/generated/reference_embeddings/mmlu_autogen_qwen
python -m zipfile -e quick_start/data/embeddings_MMLU_Autogen_qwen2.5.zip quick_start/generated/reference_embeddings/mmlu_autogen_qwen
```

This creates:

```text
quick_start/generated/reference_embeddings/mmlu_autogen_qwen/autogen_analyst_embedding_matrices_MMLU_HF_qwen2.5.pkl
quick_start/generated/reference_embeddings/mmlu_autogen_qwen/autogen_verifier_embedding_matrices_MMLU_HF_qwen2.5.pkl
quick_start/generated/reference_embeddings/mmlu_autogen_qwen/autogen_star_embedding_matrices_MMLU_HF_qwen2.5.pkl
```

## Step 1B: Re-Embed Existing Logs

From the repository root:

```bash
python quick_start/code/01_embed_reference_logs.py
```

Input:

```text
quick_start/data/conversation_logs_Math_qwen2.5.json
```

Output:

```text
quick_start/generated/embeddings/user_embedding_matrices.pkl
quick_start/generated/embeddings/assistant_embedding_matrices.pkl
```

Reference embeddings are also packaged as zip archives for inspection or reuse:

```text
quick_start/data/embeddings_Math_qwen2.5_qwen3.zip
quick_start/data/embeddings_MMLU_Autogen_qwen2.5.zip
```

These included reference embeddings use the open-source
`Qwen/Qwen3-Embedding-0.6B` model and have dimension 1024. The metadata is in:

```text
quick_start/data/embedding_metadata.json
```

Note: if logs are produced by the quick-start generation scripts, embedding is
already done during generation. This standalone embedding step is mainly for
users who bring their own conversation logs, or who want to re-embed the
included logs. It also uses `Qwen/Qwen3-Embedding-0.6B` by default and writes
`embedding_metadata.json` next to the generated embedding pickles.

Equivalent module command:

```bash
python -m matu.embed_logs \
  --logs quick_start/data/conversation_logs_Math_qwen2.5.json \
  --out_dir outputs/quick_start_embeddings \
  --roles user assistant \
  --model Qwen/Qwen3-Embedding-0.6B
```

## Step 2: MATU / CP-2

If you ran Step 1B and generated embeddings from logs, use the wrapper:

```bash
python quick_start/code/02_run_cp2_from_generated_embeddings.py
```

Input:

```text
quick_start/generated/embeddings/user_embedding_matrices.pkl
quick_start/generated/embeddings/assistant_embedding_matrices.pkl
```

Output:

```text
quick_start/generated/results/matu_scores.pkl
quick_start/generated/results/fit_dict_generated.pkl
```

Reference MATU/CP-2 result is also included:

```text
quick_start/results/fit_dict_Math_Assistonly_qwen2.5_qwen3embedding.pkl
```

If you ran Step 1A and want to run CP-2 from the extracted MATH reference
embeddings without downloading the embedding model, use the CLI directly:

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

Then continue to Step 3 with:

```bash
python quick_start/code/03_fit_to_uncertainty_generated.py
```

## Step 3: Convert Fit Dict to Uncertainty

The original CP-2 scripts save rank-wise `fit_dict` values. We convert them to
a scalar MATU uncertainty with:

```text
uncertainty = sum_R (1 - fit_R)
```

For the included reference result:

```bash
python quick_start/code/03_fit_to_uncertainty_reference.py
```

For generated Step 2 output:

```bash
python quick_start/code/03_fit_to_uncertainty_generated.py
```

Outputs:

```text
quick_start/results/uncertainty_Math_Assistonly_qwen2.5.pkl
quick_start/generated/results/uncertainty_generated.pkl
```

## Step 4: Evaluation

Evaluate the generated Step 3 uncertainty:

```bash
python quick_start/code/04_evaluate_generated_results.py
```

Evaluate the included reference uncertainty directly:

```bash
python quick_start/code/04_evaluate_reference_results.py
```

## Baseline

Evaluate the included SAUP-Multiple reference result:

```bash
python quick_start/code/05_evaluate_baselines.py
```

EigV code is provided in the root baseline script. Its result file is not
included in this quick start; it can be generated or added later:

```bash
python baselines/eigv.py \
  --logs quick_start/data/conversation_logs_Math_qwen2.5.json \
  --mode final \
  --out quick_start/generated/results/eigv_final.pkl
```
