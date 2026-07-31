"""HarmBench MATU sonuçlarını analiz et.

Tüm HarmBench deneylerinin:
  - AUROC / AUARC skorlarını
  - Safe Rate (güvenlik oranı)
  - Belirsizlik dağılımlarını (safe vs unsafe sorular)
  - Base vs Evolved karşılaştırmasını
tek bir tabloda özetler.

Kullanım:
  python analyze_harmbench_uncertainty.py
"""

import pickle
import numpy as np
from pathlib import Path

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

from matu.evaluate_uncertainty import score_from_value, accuracy_from_labels, auroc_binary, auarc


QUICK_START = Path("quick_start")

# ── HarmBench deneyleri ─────────────────────────────────────────────────────
EXPERIMENTS = [
    # (deney_adı, model_açıklama)
    ("harmbench_3b_base",              "Qwen2.5-Coder-3B-Instruct (Base)"),
    ("harmbench_3b_evolved",           "AZR-Coder-3b (Evolved)"),
    ("harmbench_7b_base",              "Qwen2.5-Coder-7B-Instruct (Base)"),
    ("harmbench_7b_evolved",           "AZR-Coder-7b (Evolved)"),
    ("harmbench_14b_base",             "Qwen2.5-Coder-14B-Instruct (Base)"),
    ("harmbench_14b_evolved",          "AZR-Coder-14b (Evolved)"),
    ("harmbench_7b_base_noncoder",     "Qwen2.5-7B-Instruct (Base, Non-Coder)"),
    ("harmbench_7b_evolved_noncoder",  "AZR-Base-7b (Evolved, Non-Coder)"),
]

# ── Base-Evolved karşılaştırma çiftleri ─────────────────────────────────────
COMPARISON_PAIRS = [
    ("harmbench_3b_base",          "harmbench_3b_evolved",          "3B Coder (Default)"),
    ("harmbench_7b_base",          "harmbench_7b_evolved",          "7B Coder (Default)"),
    ("harmbench_14b_base",         "harmbench_14b_evolved",         "14B Coder (Default)"),
    ("harmbench_7b_base_noncoder", "harmbench_7b_evolved_noncoder", "7B Non-Coder (Default)"),
]

# Add config variants dynamically
CONFIGS = [
    ("cleaned", "Cleaned"),
    ("embedding", "Embedding (4B)"),
    ("rank30", "Rank 30"),
    ("assistant_only", "Assistant Only"),
    ("user_only", "User Only"),
    ("time_exp", "Time (Exp)"),
    ("time_linear", "Time (Linear)"),
    ("time_cutoff", "Time (Cutoff)"),
    ("time_sigmoid", "Time (Sigmoid)"),
    ("time_log", "Time (Log)"),
]

for conf_key, conf_desc in CONFIGS:
    EXPERIMENTS.append((f"harmbench_3b_base_{conf_key}", f"3B Base ({conf_desc})"))
    EXPERIMENTS.append((f"harmbench_3b_evolved_{conf_key}", f"3B Evolved ({conf_desc})"))
    COMPARISON_PAIRS.append((f"harmbench_3b_base_{conf_key}", f"harmbench_3b_evolved_{conf_key}", f"3B {conf_desc}"))


def load_pickle_safe(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def analyze_experiment(exp_name: str):
    """Bir HarmBench deneyinin tüm metriklerini hesapla."""
    unc_path = QUICK_START / exp_name / "results" / "uncertainty_generated.pkl"
    acc_path = QUICK_START / exp_name / "results" / "accuracy_dict_generated.pkl"

    unc_data = load_pickle_safe(unc_path)
    acc_data = load_pickle_safe(acc_path)

    if unc_data is None or acc_data is None:
        return None

    keys = sorted(set(unc_data.keys()) & set(acc_data.keys()))
    if not keys:
        return None

    scores = np.array([score_from_value(unc_data[k], "auto") for k in keys])
    accuracies = np.array([accuracy_from_labels(acc_data[k]) for k in keys])

    # Safe/Unsafe counts
    total = len(keys)
    strict_safe = np.sum(accuracies == 1.0)
    strict_unsafe = total - strict_safe
    
    maj_safe = np.sum(accuracies >= 0.5)
    maj_unsafe = total - maj_safe

    # Safe rate (per-run level)
    all_run_scores = []
    for k in keys:
        labels = acc_data[k]
        if isinstance(labels, (list, np.ndarray)):
            all_run_scores.extend([int(x) for x in labels])
        else:
            all_run_scores.append(int(labels))
    safe_rate = np.mean(all_run_scores) * 100 if all_run_scores else 0.0

    # Uncertainty by safety
    safe_unc = scores[accuracies >= 0.5]
    unsafe_unc = scores[accuracies < 0.5]

    # AUROC (any_incorrect)
    y_error_any = (accuracies < 1.0).astype(int)
    auroc_any = auroc_binary(y_error_any, scores) if np.unique(y_error_any).size > 1 else float("nan")

    # AUROC (majority_incorrect)
    y_error_maj = (accuracies < 0.5).astype(int)
    auroc_maj = auroc_binary(y_error_maj, scores) if np.unique(y_error_maj).size > 1 else float("nan")

    # AUARC
    auarc_val = auarc(accuracies, scores)

    return {
        "total": total,
        "strict_safe": int(strict_safe),
        "strict_unsafe": int(strict_unsafe),
        "maj_safe": int(maj_safe),
        "maj_unsafe": int(maj_unsafe),
        "safe_rate": safe_rate,
        "mean_unc": float(np.mean(scores)),
        "median_unc": float(np.median(scores)),
        "mean_safe_unc": float(np.mean(safe_unc)) if len(safe_unc) > 0 else float("nan"),
        "mean_unsafe_unc": float(np.mean(unsafe_unc)) if len(unsafe_unc) > 0 else float("nan"),
        "gap": float(np.mean(unsafe_unc) - np.mean(safe_unc)) if len(safe_unc) > 0 and len(unsafe_unc) > 0 else float("nan"),
        "auroc_any": auroc_any,
        "auroc_maj": auroc_maj,
        "auarc": auarc_val,
    }


def print_header():
    print("=" * 100)
    print("HARMBENCH MATU GÜVENLIK BELİRSİZLİK ANALİZİ")
    print("=" * 100)
    print()


def print_experiment(exp_name: str, description: str, res: dict):
    print(f"--- {description} ({exp_name}) ---")
    print(f"  Toplam Behavior         : {res['total']}")
    print(f"  Safe Rate (per-run)     : {res['safe_rate']:.1f}%")
    print(f"  Safe Behavior (majority): {res['safe_behaviors']}/{res['total']}")
    print(f"  Unsafe Behavior         : {res['unsafe_behaviors']}/{res['total']}")
    print()
    print(f"  Genel Ortalama Unc.     : {res['mean_unc']:.4f}")
    print(f"  Genel Median Unc.       : {res['median_unc']:.4f}")
    print(f"  SAFE  Sorular Unc.      : {res['mean_safe_unc']:.4f}")
    print(f"  UNSAFE Sorular Unc.     : {res['mean_unsafe_unc']:.4f}")
    print(f"  GAP (Unsafe - Safe)     : {res['gap']:.4f}  <-- Büyük = İyi")
    print()
    print(f"  AUROC (any_incorrect)   : {res['auroc_any']:.4f}")
    print(f"  AUROC (majority_incorr) : {res['auroc_maj']:.4f}")
    print(f"  AUARC                   : {res['auarc']:.4f}")
    print()


def print_comparison_table(results: dict):
    """Base vs Evolved karşılaştırma tablosu."""
    print("=" * 100)
    print("BASE vs EVOLVED KARŞILAŞTIRMA TABLOSU")
    print("=" * 100)
    print()

    header = f"{'Model Çifti':<25} | {'Safe Rate B':>11} | {'Safe Rate E':>11} | {'ΔSafe':>6} | {'AUROC B':>8} | {'AUROC E':>8} | {'ΔAUROC':>7} | {'Gap B':>7} | {'Gap E':>7}"
    print(header)
    print("-" * len(header))

    for base_name, evol_name, pair_label in COMPARISON_PAIRS:
        base = results.get(base_name)
        evol = results.get(evol_name)

        if base is None and evol is None:
            print(f"{pair_label:<25} | {'N/A':>11} | {'N/A':>11} | {'N/A':>6} | {'N/A':>8} | {'N/A':>8} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7}")
            continue

        sr_b = f"{base['safe_rate']:.1f}%" if base else "N/A"
        sr_e = f"{evol['safe_rate']:.1f}%" if evol else "N/A"
        d_sr = f"{evol['safe_rate'] - base['safe_rate']:+.1f}" if base and evol else "N/A"

        au_b = f"{base['auroc_any']:.4f}" if base and not np.isnan(base['auroc_any']) else "N/A"
        au_e = f"{evol['auroc_any']:.4f}" if evol and not np.isnan(evol['auroc_any']) else "N/A"
        d_au = f"{evol['auroc_any'] - base['auroc_any']:+.4f}" if base and evol and not np.isnan(base['auroc_any']) and not np.isnan(evol['auroc_any']) else "N/A"

        gp_b = f"{base['gap']:.4f}" if base and not np.isnan(base['gap']) else "N/A"
        gp_e = f"{evol['gap']:.4f}" if evol and not np.isnan(evol['gap']) else "N/A"

        print(f"{pair_label:<25} | {sr_b:>11} | {sr_e:>11} | {d_sr:>6} | {au_b:>8} | {au_e:>8} | {d_au:>7} | {gp_b:>7} | {gp_e:>7}")

    print()


def print_summary_table(results: dict):
    """Tüm deneyleri tek bir tabloda özetle."""
    print("=" * 100)
    print("ÖZET TABLO")
    print("=" * 100)
    print()

    header = f"{'Deney':<45} | {'Safe%':>6} | {'#Safe':>5} | {'#Unsafe':>7} | {'Safe Unc':>8} | {'Unsafe Unc':>10} | {'AUROC':>7} | {'AUARC':>7} | {'Gap':>7}"
    print(header)
    print("-" * len(header))

    for exp_name, desc in EXPERIMENTS:
        res = results.get(exp_name)
        if res is None:
            print(f"{desc:<45} | {'N/A':>6} | {'N/A':>5} | {'N/A':>7} | {'N/A':>8} | {'N/A':>10} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7}")
            continue

        auroc_str = f"{res['auroc_any']:.4f}" if not np.isnan(res['auroc_any']) else "N/A"
        auarc_str = f"{res['auarc']:.4f}" if not np.isnan(res['auarc']) else "N/A"
        gap_str = f"{res['gap']:.4f}" if not np.isnan(res['gap']) else "N/A"
        mean_unc_str = f"{res['mean_unc']:.4f}" if not np.isnan(res['mean_unc']) else "N/A"
        median_unc_str = f"{res['median_unc']:.4f}" if not np.isnan(res['median_unc']) else "N/A"
        safe_unc_str = f"{res['mean_safe_unc']:.4f}" if not np.isnan(res['mean_safe_unc']) else "N/A"
        unsafe_unc_str = f"{res['mean_unsafe_unc']:.4f}" if not np.isnan(res['mean_unsafe_unc']) else "N/A"
        
        print(f"{desc:<45} | {res['safe_rate']:>5.1f}% | {res['maj_safe']:>5} | {res['maj_unsafe']:>7} | {mean_unc_str:>8} | {median_unc_str:>10} | {safe_unc_str:>8} | {unsafe_unc_str:>10} | {auroc_str:>7} | {auarc_str:>7} | {gap_str:>7}")

    print()
    
    # --- CSV Export ---
    csv_path = Path("MATU_Results.csv")
    with csv_path.open("w", encoding="utf-8") as f:
        # Header
        f.write("Model,Strict_Total,Strict_Safe_Rate,Strict_AUROC,Strict_AUARC,Relaxed_Total,Relaxed_Safe_Rate,Relaxed_AUROC,Relaxed_AUARC,Mean_Unc,Median_Unc,Safe_Unc,Unsafe_Unc,Gap\n")
        
        for exp_name, desc in EXPERIMENTS:
            res = results.get(exp_name)
            if res is None:
                continue
                
            strict_total = res['total']
            strict_safe_rate = res['strict_safe'] / strict_total if strict_total > 0 else 0.0
            
            relaxed_total = res['total']
            relaxed_safe_rate = res['maj_safe'] / relaxed_total if relaxed_total > 0 else 0.0
            
            strict_auroc = f"{res['auroc_any']:.4f}" if not np.isnan(res['auroc_any']) else ""
            strict_auarc = f"{res['auarc']:.4f}" if not np.isnan(res['auarc']) else ""
            
            relaxed_auroc = f"{res['auroc_maj']:.4f}" if not np.isnan(res['auroc_maj']) else ""
            relaxed_auarc = f"{res['auarc']:.4f}" if not np.isnan(res['auarc']) else "" # AUARC is the same
            
            mean_unc = f"{res['mean_unc']:.4f}" if not np.isnan(res['mean_unc']) else ""
            median_unc = f"{res['median_unc']:.4f}" if not np.isnan(res['median_unc']) else ""
            safe_unc = f"{res['mean_safe_unc']:.4f}" if not np.isnan(res['mean_safe_unc']) else ""
            unsafe_unc = f"{res['mean_unsafe_unc']:.4f}" if not np.isnan(res['mean_unsafe_unc']) else ""
            gap = f"{res['gap']:.4f}" if not np.isnan(res['gap']) else ""
            
            f.write(f"{desc},{strict_total},{strict_safe_rate:.4f},{strict_auroc},{strict_auarc},{relaxed_total},{relaxed_safe_rate:.4f},{relaxed_auroc},{relaxed_auarc},{mean_unc},{median_unc},{safe_unc},{unsafe_unc},{gap}\n")
            
    print(f"\nSonuclar MATU_Results.csv dosyasina kaydedildi! Bu dosyayi bilgisayarina indirip Excel'de acabilirsin.")


def main():
    print_header()

    results = {}
    for exp_name, description in EXPERIMENTS:
        res = analyze_experiment(exp_name)
        if res is not None:
            results[exp_name] = res
            print_experiment(exp_name, description, res)
        else:
            print(f"--- {description} ({exp_name}) ---")
            print("  (Henüz veriler hazır değil veya klasör yok)\n")

    if results:
        print_summary_table(results)
        print_comparison_table(results)

    # Genel yorum
    print("=" * 100)
    print("YORUMLAMA REHBERİ")
    print("=" * 100)
    print("""
  Safe Rate ↓  = Model evrimle güvenlik kaybediyor (safety drift)
  AUROC ↑      = MATU belirsizliği unsafe çıktıları iyi tahmin ediyor
  Gap ↑        = MATU unsafe soruları safe sorulardan iyi ayırıyor

  İdeal Senaryo:
    - Base modelde yüksek Safe Rate + düşük AUROC (hep safe → ayrım yok)
    - Evolved modelde düşük Safe Rate + yüksek AUROC (safety drift var ama MATU tahmin edebiliyor)
    → MATU'nun güvenlik guardrail olarak kullanılabileceğini gösterir
""")


if __name__ == "__main__":
    main()
