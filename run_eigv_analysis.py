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
            print(f"Uyari: {model_name} icin log dosyasi bulunamadi: {log_path}")
            return None
            
        print(f"[{model_name}] EigV (NLI) skorlari hesaplaniyor (bu islem biraz surebilir)...")
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
            print(f"[{model_name}] EigV hesaplamasi basariyla bitti.\n")
        except subprocess.CalledProcessError as e:
            print(f"Hata: {model_name} icin EigV hesaplanirken bir sorun olustu:\n{e}\n")
            return None

    if not acc_path.exists():
        print(f"Uyari: {model_name} icin accuracy (dogruluk) dosyasi bulunamadi: {acc_path}")
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
    models = ["gen_3b_base", "gen_3b_evolved", "gen_7b_base", "gen_7b_evolved", "gen_14b_base", "gen_14b_evolved"]
    print("="*75)
    print("EIGV (NLI) BELIRSIZLIK (UNCERTAINTY AMOUNT) ANALIZI")
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
            print("(Veriler hazir degil veya hesaplama basarisiz)\n")
            
if __name__ == "__main__":
    main()
