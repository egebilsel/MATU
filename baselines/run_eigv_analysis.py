import os
import subprocess
import pickle
import numpy as np
from pathlib import Path
from matu.evaluate_uncertainty import score_from_value, accuracy_from_labels

def get_detailed_uncertainty(model_name):
    # Log and Output Paths
    log_path = Path(f"quick_start/{model_name}/conversation_logs_hf_qwen.json")
    unc_path = Path(f"quick_start/{model_name}/results/eigv_uncertainty.pkl")
    acc_path = Path(f"quick_start/{model_name}/results/accuracy_dict_generated.pkl")
    
    # If the EigV pkl does not exist, calculate it using baselines/eigv.py
    if not unc_path.exists():
        if not log_path.exists():
            print(f"Warning: Log file not found for {model_name}: {log_path}")
            return None
            
        print(f"[{model_name}] Calculating EigV (NLI) scores (this might take a while)...")
        unc_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "python", "baselines/eigv.py",
            "--logs", str(log_path),
            "--mode", "final",
            "--out", str(unc_path)
        ]
        
        # Run the subprocess
        try:
            subprocess.run(cmd, check=True)
            print(f"[{model_name}] EigV calculation completed successfully.\n")
        except subprocess.CalledProcessError as e:
            print(f"Error: A problem occurred while calculating EigV for {model_name}:\n{e}\n")
            return None

    if not acc_path.exists():
        print(f"Warning: Accuracy file not found for {model_name}: {acc_path}")
        return None
        
    # Load the results
    with open(unc_path, "rb") as f:
        unc_data = pickle.load(f)
    with open(acc_path, "rb") as f:
        acc_data = pickle.load(f)
        
    correct_scores = []
    incorrect_scores = []
    all_scores = []
    
    # Intersection of keys to be safe
    keys = set(unc_data.keys()) & set(acc_data.keys())
    
    for k in keys:
        # score_from_value can parse the EigV dict automatically using mode="auto"
        score = score_from_value(unc_data[k], "auto")
        acc = accuracy_from_labels(acc_data[k])
        
        all_scores.append(score)
        # Assuming majority_incorrect logic: >= 0.5 accuracy is correct
        if acc >= 0.5:
            correct_scores.append(score)
        else:
            incorrect_scores.append(score)
            
    return {
        "mean_all": np.mean(all_scores) if all_scores else 0,
        "median_all": np.median(all_scores) if all_scores else 0,
        "mean_correct": np.mean(correct_scores) if correct_scores else 0,
        "mean_incorrect": np.mean(incorrect_scores) if incorrect_scores else 0,
        "correct_count": len(correct_scores),
        "incorrect_count": len(incorrect_scores)
    }

def main():
    models = ["gen_7b_base_noncoder", "gen_7b_evolved_noncoder"]
    print("="*75)
    print("EIGV (NLI) UNCERTAINTY AMOUNT ANALYSIS")
    print("="*75)
    
    for model in models:
        res = get_detailed_uncertainty(model)
        if res is not None:
            gap = res['mean_incorrect'] - res['mean_correct']
            print(f"--- {model.upper()} ---")
            print(f"Overall Median Uncertainty    : {res['median_all']:.4f}")
            print(f"Overall Mean Uncertainty      : {res['mean_all']:.4f}")
            print(f"CORRECT Questions Uncertainty ({res['correct_count']:>3} questions): {res['mean_correct']:.4f} (Expected: Low)")
            print(f"INCORRECT Questions Uncertain.({res['incorrect_count']:>3} questions): {res['mean_incorrect']:.4f} (Expected: High)")
            print(f"GAP (Incorrect - Correct)     : {gap:.4f}  <-- Higher is better!\n")
        else:
            print(f"--- {model.upper()} ---")
            print("(Data not ready or calculation failed)\n")
            
if __name__ == "__main__":
    main()
