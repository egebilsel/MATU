import re
import pandas as pd
import io

def update_markdown_and_excel():
    with open('harmbench_results_summary.md', 'r') as f:
        md_text = f.read()
        
    # Extract data from text blocks
    blocks = md_text.split('--- ')
    data = []
    
    for block in blocks[1:]:
        if '===' in block:
            break
            
        lines = block.split('\n')
        name_line = lines[0].strip()
        if not name_line:
            continue
            
        exp_name = name_line.split('(')[-1].replace(')','').strip('- ')
        desc = name_line.split('(')[0].strip()
        if '(Base' in name_line or '(Evolved' in name_line:
            desc = name_line.rsplit('(', 1)[0].strip()
            
        # Vars
        safe_rate = 0.0
        n_safe = 0
        n_unsafe = 0
        mean_unc = "N/A"
        median_unc = "N/A"
        safe_unc = "N/A"
        unsafe_unc = "N/A"
        gap = "N/A"
        auroc = "N/A"
        auarc = "N/A"
        
        for line in lines:
            if 'Safe Rate (per-run)' in line:
                safe_rate = line.split(':')[1].strip().replace('%', '')
            elif 'Safe Behavior (majority)' in line:
                n_safe = line.split(':')[1].split('/')[0].strip()
            elif 'Unsafe Behavior' in line:
                n_unsafe = line.split(':')[1].split('/')[0].strip()
            elif 'Genel Ortalama Unc.' in line:
                mean_unc = line.split(':')[1].strip()
            elif 'Genel Median Unc.' in line:
                median_unc = line.split(':')[1].strip()
            elif 'SAFE  Sorular Unc.' in line:
                safe_unc = line.split(':')[1].strip()
            elif 'UNSAFE Sorular Unc.' in line:
                unsafe_unc = line.split(':')[1].strip()
            elif 'GAP (Unsafe - Safe)' in line:
                gap = line.split(':')[1].split('<')[0].strip()
            elif 'AUROC (any_incorrect)' in line:
                auroc = line.split(':')[1].strip()
            elif 'AUARC' in line:
                auarc = line.split(':')[1].strip()
                
        # Base/Evolved handling for desc
        if 'Base' in name_line and 'Non-Coder' not in name_line:
            desc = desc.replace('Qwen2.5-Coder-', '')
        
        row = f"{desc:<45} | {float(safe_rate):>5.1f}% | {n_safe:>5} | {n_unsafe:>7} | {mean_unc:>8} | {median_unc:>10} | {safe_unc:>8} | {unsafe_unc:>10} | {auroc:>7} | {auarc:>7} | {gap:>7}"
        data.append(row)
        
    # Rebuild the table
    header = f"{'Deney':<45} | {'Safe%':>6} | {'#Safe':>5} | {'#Unsafe':>7} | {'Mean Unc':>8} | {'Median Unc':>10} | {'Safe Unc':>8} | {'Unsafe Unc':>10} | {'AUROC':>7} | {'AUARC':>7} | {'Gap':>7}"
    sep = "-" * len(header)
    
    new_table = f"====================================================================================================\nÖZET TABLO\n====================================================================================================\n\n{header}\n{sep}\n"
    for row in data:
        new_table += row + "\n"
        
    # Replace in markdown
    start_idx = md_text.find("====================================================================================================\nÖZET TABLO")
    end_idx = md_text.find("====================================================================================================\nBASE vs EVOLVED")
    
    new_md = md_text[:start_idx] + new_table + "\n" + md_text[end_idx:]
    
    with open('harmbench_results_summary.md', 'w') as f:
        f.write(new_md)
        
if __name__ == '__main__':
    update_markdown_and_excel()
