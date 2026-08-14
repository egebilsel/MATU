import os

MODELS = ["base", "evolved"]

CONFIGS = [
    {"name": "cleaned", "roles": "user assistant", "model": "Qwen/Qwen3-Embedding-0.6B", "max_rank": 50, "time": None, "clean": True},
    {"name": "embedding", "roles": "user assistant", "model": "Qwen/Qwen3-Embedding-4B", "max_rank": 50, "time": None, "clean": False},
    {"name": "rank30", "roles": "user assistant", "model": "Qwen/Qwen3-Embedding-0.6B", "max_rank": 30, "time": None, "clean": False},
    {"name": "assistant_only", "roles": "assistant", "model": "Qwen/Qwen3-Embedding-0.6B", "max_rank": 50, "time": None, "clean": False},
    {"name": "user_only", "roles": "user", "model": "Qwen/Qwen3-Embedding-0.6B", "max_rank": 50, "time": None, "clean": False},
    {"name": "time_exp", "roles": "user assistant", "model": "Qwen/Qwen3-Embedding-0.6B", "max_rank": 50, "time": "exp", "clean": False},
    {"name": "time_linear", "roles": "user assistant", "model": "Qwen/Qwen3-Embedding-0.6B", "max_rank": 50, "time": "linear", "clean": False},
    {"name": "time_cutoff", "roles": "user assistant", "model": "Qwen/Qwen3-Embedding-0.6B", "max_rank": 50, "time": "cutoff", "clean": False},
    {"name": "time_sigmoid", "roles": "user assistant", "model": "Qwen/Qwen3-Embedding-0.6B", "max_rank": 50, "time": "sigmoid", "clean": False},
    {"name": "time_log", "roles": "user assistant", "model": "Qwen/Qwen3-Embedding-0.6B", "max_rank": 50, "time": "log", "clean": False},
    {"name": "assistant_embedding", "roles": "assistant", "model": "Qwen/Qwen3-Embedding-4B", "max_rank": 50, "time": None, "clean": False},
]

TEMPLATE = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J hb-3b_{short_model}_{config_name}
#SBATCH --chdir=/dl_scratch2/ege/ege/repos/MATU
#SBATCH --nodelist=adep2
#SBATCH --gres=gpu:1
#SBATCH --output=/dl_scratch2/ege/ege/repos/MATU/results/harmbench/harmbench_3b_{model}_{config_name}-%j.out
#SBATCH --error=/dl_scratch2/ege/ege/repos/MATU/results/harmbench/harmbench_3b_{model}_{config_name}-%j.err
#SBATCH --time=2-00:00:00

# 1. Environment ve Cache Ayarları
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /dl_scratch2/ege/ege/envs/matu-uq/

export HF_HOME=/dl_scratch2/ege/hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /dl_scratch2/ege/ege/repos/MATU

# Çıktı klasörlerini oluştur
mkdir -p quick_start/harmbench/harmbench_3b_{model}_{config_name}/embeddings
mkdir -p quick_start/harmbench/harmbench_3b_{model}_{config_name}/results

# Sadece önceki log dosyasını kopyalıyoruz (Yeniden inference yapmamak için)
cp quick_start/harmbench/harmbench_3b_{model}/conversation_logs_harmbench.json quick_start/harmbench/harmbench_3b_{model}_{config_name}/
{cleaner_step}
echo ""
echo "=== Step 01: Embed Logs (GPU) ==="
python -m matu.embed_logs \\
  --logs quick_start/harmbench/harmbench_3b_{model}_{config_name}/{log_file} \\
  --out_dir quick_start/harmbench/harmbench_3b_{model}_{config_name}/embeddings \\
  --roles {roles} \\
  --model "{embed_model}"

echo ""
echo "=== Step 02: CP-2 Tensor Decomposition (CPU) ==="
python quick_start/code/02_run_cp2_from_generated_embeddings.py \\
  --embedding-source harmbench/harmbench_3b_{model}_{config_name} \\
  --max_rank {max_rank}{time_arg}

echo ""
echo "=== Step 03: Fit to Uncertainty ==="
python quick_start/code/03_fit_to_uncertainty_generated.py harmbench/harmbench_3b_{model}_{config_name}

echo ""
echo "=== Step 04: Grade Safety ==="
python quick_start/code/grade_harmbench.py quick_start/harmbench/harmbench_3b_{model}_{config_name} \\
  --judge keyword

echo "--- EVALUATION: ANY UNSAFE (STRICT) ---"
python -m matu.evaluate_uncertainty \\
  --uncertainty quick_start/harmbench/harmbench_3b_{model}_{config_name}/results/uncertainty_generated.pkl \\
  --labels quick_start/harmbench/harmbench_3b_{model}_{config_name}/results/accuracy_dict_generated.pkl \\
  --score_mode raw \\
  --error_rule any_incorrect

echo ""
echo "--- EVALUATION: MAJORITY UNSAFE (RELAXED) ---"
python -m matu.evaluate_uncertainty \\
  --uncertainty quick_start/harmbench/harmbench_3b_{model}_{config_name}/results/uncertainty_generated.pkl \\
  --labels quick_start/harmbench/harmbench_3b_{model}_{config_name}/results/accuracy_dict_generated.pkl \\
  --score_mode raw \\
  --error_rule majority_incorrect

echo "HARMBENCH MATU PIPELINE DONE FOR 3b_{model}_{config_name}!"
"""

for model in MODELS:
    short_model = "b" if model == "base" else "e"
    for conf in CONFIGS:
        config_name = conf["name"]
        
        log_file = "conversation_logs_harmbench.json"
        cleaner_step = ""
        
        if conf["clean"]:
            log_file = "cleaned_conversation_logs.json"
            cleaner_step = f"""
echo "=== Step 00.5: Remove Stopwords with LLM (GPU) ==="
python -m matu.remove_stopwords_llm \\
  --input_logs quick_start/harmbench_3b_{model}/conversation_logs_harmbench.json \\
  --output_logs quick_start/harmbench_3b_{model}_{config_name}/cleaned_conversation_logs.json \\
  --model "Qwen/Qwen2.5-14B-Instruct"
"""
        
        time_arg = f" \\\n  --time-weighting {conf['time']}" if conf["time"] else ""
        
        content = TEMPLATE.format(
            model=model,
            short_model=short_model,
            config_name=config_name,
            cleaner_step=cleaner_step,
            log_file=log_file,
            roles=conf["roles"],
            embed_model=conf["model"],
            max_rank=conf["max_rank"],
            time_arg=time_arg
        )
        
        filename = f"run_harmbench_3b_{model}_{config_name}.slurm"
        with open(filename, "w") as f:
            f.write(content)
        print(f"Created {filename}")

print("All HarmBench 3B config SLURMs created successfully.")
