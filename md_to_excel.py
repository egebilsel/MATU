import pandas as pd
import io

def parse_markdown_table(md_text, section_header):
    lines = md_text.split('\n')
    start_idx = -1
    for i, line in enumerate(lines):
        if section_header in line:
            # Bulunan başlıktan sonraki tabloyu bul
            for j in range(i+1, len(lines)):
                if '|' in lines[j] and '---' in lines[j+1]:
                    start_idx = j
                    break
            if start_idx != -1:
                break
                
    if start_idx == -1:
        return None
        
    table_lines = []
    # Header
    table_lines.append(lines[start_idx].strip())
    # Skip separator
    # Data rows
    for j in range(start_idx+2, len(lines)):
        line = lines[j].strip()
        if not line or '|' not in line:
            break
        table_lines.append(line)
        
    if not table_lines:
        return None
        
    # Markdown formatından temizle
    cleaned_lines = []
    for line in table_lines:
        cleaned_lines.append('\t'.join([col.strip() for col in line.strip('|').split('|')]))
        
    csv_io = io.StringIO('\n'.join(cleaned_lines))
    df = pd.read_csv(csv_io, sep='\t')
    return df

def main():
    with open('harmbench_results_summary.md', 'r') as f:
        md_text = f.read()
        
    df_summary = parse_markdown_table(md_text, 'ÖZET TABLO')
    df_compare = parse_markdown_table(md_text, 'BASE vs EVOLVED KARŞILAŞTIRMA TABLOSU')
    
    with pd.ExcelWriter('MATU_Results.xlsx', engine='openpyxl') as writer:
        if df_summary is not None:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
        if df_compare is not None:
            df_compare.to_excel(writer, sheet_name='Comparison', index=False)
            
    print("Excel dosyası başarıyla oluşturuldu: MATU_Results.xlsx")

if __name__ == '__main__':
    main()
