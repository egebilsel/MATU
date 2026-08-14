import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matu.evaluate_uncertainty import score_from_value

QUICK_START = Path("quick_start/harmbench")

def get_data_dict(exp_name):
    unc_path = QUICK_START / exp_name / "results" / "uncertainty_generated.pkl"
    acc_path = QUICK_START / exp_name / "results" / "accuracy_dict_generated.pkl"

    if not unc_path.exists() or not acc_path.exists():
        print(f"Data not found for {exp_name}")
        return {}

    with unc_path.open("rb") as f:
        unc_data = pickle.load(f)
    with acc_path.open("rb") as f:
        acc_data = pickle.load(f)

    keys = sorted(set(unc_data.keys()) & set(acc_data.keys()))

    result = {}
    for k in keys:
        labels = acc_data[k]
        unc_val = unc_data[k]
        
        if isinstance(labels, (list, np.ndarray)):
            safety = np.mean([int(x) for x in labels])
        else:
            safety = float(labels)
            
        unc = score_from_value(unc_val, "auto")
        result[k] = {"safety": safety, "uncertainty": unc}
        
    return result

pairs = [
    ("3B Coder", "harmbench_3b_base", "harmbench_3b_evolved"),
    ("7B Coder", "harmbench_7b_base", "harmbench_7b_evolved"),
    ("14B Coder", "harmbench_14b_base", "harmbench_14b_evolved"),
    ("7B Non-Coder", "harmbench_7b_base_noncoder", "harmbench_7b_evolved_noncoder")
]

for title, base_exp, evolved_exp in pairs:
    base_data = get_data_dict(base_exp)
    evolved_data = get_data_dict(evolved_exp)
    
    common_keys = set(base_data.keys()) & set(evolved_data.keys())
    
    delta_safeties = []
    delta_uncertainties = []
    
    for k in common_keys:
        d_safety = evolved_data[k]["safety"] - base_data[k]["safety"]
        d_unc = evolved_data[k]["uncertainty"] - base_data[k]["uncertainty"]
        
        delta_safeties.append(d_safety)
        delta_uncertainties.append(d_unc)
        
    if not delta_safeties:
        continue
        
    plt.figure(figsize=(10, 6))
    
    # Adding jitter to both axes slightly since deltas might overlap exactly (like 0,0)
    jittered_safeties = np.array(delta_safeties) + np.random.normal(0, 0.015, len(delta_safeties))
    jittered_uncertainties = np.array(delta_uncertainties) + np.random.normal(0, 0.015, len(delta_uncertainties))
    
    plt.scatter(jittered_safeties, jittered_uncertainties, color="purple", alpha=0.5)
    
    # Add horizontal and vertical lines at 0 to easily see quadrants
    plt.axhline(0, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
    
    plt.title(f"MATU Deltas (Evolved - Base) ({title})")
    plt.xlabel("Δ Safety (Evolved - Base)")
    plt.ylabel("Δ Uncertainty (Evolved - Base)")
    plt.grid(True, alpha=0.3)

    out_file = f"MATU_delta_{title.replace(' ', '_').lower()}.png"
    plt.savefig(out_file)
    plt.close()
    print(f"Saved {out_file}")
