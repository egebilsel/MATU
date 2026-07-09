import pickle
import numpy as np
from pathlib import Path
from matu.evaluate_uncertainty import score_from_value

def get_mean_uncertainty(model_name):
    path = Path(f"quick_start/{model_name}/results/uncertainty_generated.pkl")
    if not path.exists():
        return None
        
    with open(path, "rb") as f:
        data = pickle.load(f)
        
    # score_from_value fonksiyonunu kullanarak (MATU'nun orijinal hesaplama yontemi) degerleri cikar
    scores = [score_from_value(val, "raw") for val in data.values()]
    return np.mean(scores)

def main():
    models = ["gen_3b_base", "gen_3b_evolved", "gen_7b_base", "gen_7b_evolved"]
    print("="*50)
    print("ORTALAMA BELIRSIZLIK (UNCERTAINTY AMOUNT) ANALIZI")
    print("="*50)
    
    for model in models:
        mean_unc = get_mean_uncertainty(model)
        if mean_unc is not None:
            print(f"{model:<20}: {mean_unc:.4f}")
        else:
            print(f"{model:<20}: (Henuz tamamlanmadi)")
            
    print("="*50)
    print("NOT: Sayi ne kadar YUKSEKSE, model ortalamada o kadar kararsiz/suphelidir.")
    print("Sayi ne kadar DUSUKSE, model ortalamada o kadar kendinden emin (fikri sabit) demektir.")

if __name__ == "__main__":
    main()
