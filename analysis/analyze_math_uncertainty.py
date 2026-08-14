import pickle
import numpy as np
from pathlib import Path
from matu.evaluate_uncertainty import score_from_value, accuracy_from_labels

QUICK_START = Path(__file__).resolve().parents[1] / "quick_start" / "math"

def get_detailed_uncertainty(model_name):
    if model_name == "Math_qwen2.5_baseline":
        unc_path = QUICK_START / "results/uncertainty_Math_qwen2.5.pkl"
        acc_path = QUICK_START / "results/accuracy_dict_Math_qwen2.5.pkl"
    else:
        unc_path = QUICK_START / f"{model_name}/results/uncertainty_generated.pkl"
        acc_path = QUICK_START / f"{model_name}/results/accuracy_dict_generated.pkl"
    
    if not unc_path.exists() or not acc_path.exists():
        return None
        
    with open(unc_path, "rb") as f:
        unc_data = pickle.load(f)
    with open(acc_path, "rb") as f:
        acc_data = pickle.load(f)
        
    correct_scores = []
    incorrect_scores = []
    all_scores = []
    
    keys = set(unc_data.keys()) & set(acc_data.keys())
    
    for k in keys:
        score = score_from_value(unc_data[k], "auto")
        acc = accuracy_from_labels(acc_data[k])
        
        all_scores.append(score)
        # according to majority_incorrect rule: count as Correct if score is 5 or above out of 10
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
    models = ["Math_qwen2.5_baseline", "gen_3b_base", "gen_3b_evolved", "gen_3b_base_embedding", "gen_3b_evolved_embedding", "gen_3b_base_cleaned", "gen_3b_evolved_cleaned", "gen_3b_base_rank30", "gen_3b_evolved_rank30", "gen_3b_base_time_exp", "gen_3b_evolved_time_exp", "gen_3b_base_time_linear", "gen_3b_evolved_time_linear", "gen_3b_base_time_cutoff", "gen_3b_evolved_time_cutoff", "gen_3b_base_time_sigmoid", "gen_3b_evolved_time_sigmoid", "gen_3b_base_time_log", "gen_3b_evolved_time_log", "gen_3b_base_assistant_only", "gen_3b_evolved_assistant_only", "gen_3b_base_time_exp_assistant_only", "gen_3b_evolved_time_exp_assistant_only", "gen_3b_base_assistant_embedding", "gen_3b_evolved_assistant_embedding", "gen_3b_base_time_exp_assistant_embedding", "gen_3b_evolved_time_exp_assistant_embedding", "gen_3b_base_user_only", "gen_3b_evolved_user_only", "gen_3b_base_cleaned_4b", "gen_3b_evolved_cleaned_4b", "gen_7b_base", "gen_7b_evolved", "gen_7b_base_noncoder", "gen_7b_evolved_noncoder", "gen_14b_base", "gen_14b_evolved"]
    print("="*75)
    print("DETAILED UNCERTAINTY AMOUNT ANALYSIS")
    print("="*75)
    
    for model in models:
        res = get_detailed_uncertainty(model)
        if res is not None:
            gap = res['mean_incorrect'] - res['mean_correct']
            print(f"--- {model.upper()} ---")
            print(f"Overall Median Uncertainty      : {res['median_all']:.4f}")
            print(f"Overall Mean Uncertainty        : {res['mean_all']:.4f}")
            print(f"CORRECT Questions Uncertainty   ({res['correct_count']:>3} questions): {res['mean_correct']:.4f} (Expected: Low)")
            print(f"INCORRECT Questions Uncertainty ({res['incorrect_count']:>3} questions): {res['mean_incorrect']:.4f} (Expected: High)")
            print(f"GAP (Incorrect - Correct)       : {gap:.4f}  <-- The higher this number, the better!\n")
        else:
            print(f"--- {model.upper()} ---")
            print("(Data not ready or folder does not exist)\n")
            
if __name__ == "__main__":
    main()
