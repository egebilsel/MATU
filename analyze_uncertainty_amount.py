import pickle
import numpy as np
from pathlib import Path
from matu.evaluate_uncertainty import score_from_value, accuracy_from_labels

def get_detailed_uncertainty(model_name):
    if model_name == "Math_qwen2.5_baseline":
        unc_path = Path("quick_start/results/uncertainty_Math_qwen2.5.pkl")
        acc_path = Path("quick_start/results/accuracy_dict_Math_qwen2.5.pkl")
    else:
        unc_path = Path(f"quick_start/{model_name}/results/uncertainty_generated.pkl")
        acc_path = Path(f"quick_start/{model_name}/results/accuracy_dict_generated.pkl")
    
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
        # majority_incorrect kuralina gore: 10 uzerinden 5 ve ustu yapanlari Dogru sayiyoruz
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
    models = ["Math_qwen2.5_baseline", "gen_3b_base", "gen_3b_evolved", "gen_3b_base_embedding", "gen_3b_evolved_embedding", "gen_7b_base", "gen_7b_evolved", "gen_7b_base_noncoder", "gen_7b_evolved_noncoder", "gen_14b_base", "gen_14b_evolved"]
    print("="*75)
    print("DETAYLI BELIRSIZLIK (UNCERTAINTY AMOUNT) ANALIZI")
    print("="*75)
    
    for model in models:
        res = get_detailed_uncertainty(model)
        if res is not None:
            gap = res['mean_incorrect'] - res['mean_correct']
            print(f"--- {model.upper()} ---")
            print(f"Genel Median Belirsizlik      : {res['median_all']:.4f}")
            print(f"Genel Ortalama Belirsizlik    : {res['mean_all']:.4f}")
            print(f"DOGRU Sorularin Belirsizligi  ({res['correct_count']:>3} soru): {res['mean_correct']:.4f} (Beklenen: Dusuk)")
            print(f"YANLIS Sorularin Belirsizligi ({res['incorrect_count']:>3} soru): {res['mean_incorrect']:.4f} (Beklenen: Yuksek)")
            print(f"FARK (Yanlis - Dogru Gap)     : {gap:.4f}  <-- Bu sayi ne kadar buyukse o kadar iyi!\n")
        else:
            print(f"--- {model.upper()} ---")
            print("(Henuz veriler hazir degil veya klasor yok)\n")
            
if __name__ == "__main__":
    main()
