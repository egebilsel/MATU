import json, re, pickle
from pathlib import Path

logs_path = Path("quick_start/generated/conversation_logs_hf_qwen.json")
math_data_dir = Path("quick_start/data/MATH/test")
out_path = Path("quick_start/generated/results/accuracy_dict_generated.pkl")

print("Loglar okunuyor...")
with open(logs_path, "r", encoding="utf-8") as f:
    logs = json.load(f)

accuracy_dict = {}

def get_boxed_answer(text):
    match = re.search(r'\\boxed{([^}]+)}', text)
    return match.group(1) if match else text.split()[-1]

for problem_id, runs in logs.items():
    parts = problem_id.split('_')
    cat, file_id = "_".join(parts[:-1]), parts[-1]
    
    with open(math_data_dir / cat / f"{file_id}.json", "r") as f:
        truth_data = json.load(f)
    
    truth_answer = get_boxed_answer(truth_data["solution"])
    
    run_scores = []
    for run in runs:
        # HATA BURADAYDI, DÜZELTİLDİ: "output"
        final_msg = run[-1]["output"] if run else ""
        
        if truth_answer in final_msg:
            run_scores.append(1)
        else:
            run_scores.append(0)
            
    accuracy_dict[problem_id] = run_scores
    print(f"Soru: {problem_id} | Doğru Bilme Sayısı: {sum(run_scores)}/10")

out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "wb") as f:
    pickle.dump(accuracy_dict, f)

print(f"\nBaşarılı! accuracy_dict dosyası şuraya kaydedildi: {out_path}")
