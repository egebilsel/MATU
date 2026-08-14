import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matu.evaluate_uncertainty import score_from_value

QUICK_START = Path("quick_start/harmbench")

def get_data(exp_name):
    unc_path = QUICK_START / exp_name / "results" / "uncertainty_generated.pkl"
    acc_path = QUICK_START / exp_name / "results" / "accuracy_dict_generated.pkl"

    if not unc_path.exists() or not acc_path.exists():
        print(f"Data not found for {exp_name}")
        return [], []

    with unc_path.open("rb") as f:
        unc_data = pickle.load(f)
    with acc_path.open("rb") as f:
        acc_data = pickle.load(f)

    keys = sorted(set(unc_data.keys()) & set(acc_data.keys()))

    safeties = []
    uncertainties = []

    for k in keys:
        labels = acc_data[k]
        unc_val = unc_data[k]
        
        if isinstance(labels, (list, np.ndarray)):
            safety = np.mean([int(x) for x in labels])
        else:
            safety = float(labels)
            
        safeties.append(safety)
        unc = score_from_value(unc_val, "auto")
        uncertainties.append(unc)
        
    return safeties, uncertainties


pairs = [
    ("3B Coder", "harmbench_3b_base", "harmbench_3b_evolved"),
    ("7B Coder", "harmbench_7b_base", "harmbench_7b_evolved"),
    ("14B Coder", "harmbench_14b_base", "harmbench_14b_evolved"),
    ("7B Non-Coder", "harmbench_7b_base_noncoder", "harmbench_7b_evolved_noncoder")
]

for title, base_exp, evolved_exp in pairs:
    plt.figure(figsize=(10, 6))

    # Base Model
    safeties_base, uncertainties_base = get_data(base_exp)
    if safeties_base:
        jittered_safeties_base = np.array(safeties_base) + np.random.normal(0, 0.02, len(safeties_base))
        plt.scatter(jittered_safeties_base, uncertainties_base, color="blue", label="Base Model", alpha=0.5)

    # Evolved Model
    safeties_evolved, uncertainties_evolved = get_data(evolved_exp)
    if safeties_evolved:
        jittered_safeties_evolved = np.array(safeties_evolved) + np.random.normal(0, 0.02, len(safeties_evolved))
        plt.scatter(jittered_safeties_evolved, uncertainties_evolved, color="orange", label="Evolved Model", alpha=0.5)

    plt.title(f"Safety vs Uncertainty ({title})")
    plt.xlabel("Safety (Jittered)")
    plt.ylabel("Uncertainty")
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_file = f"scatter_{title.replace(' ', '_').lower()}.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")

