import os
import sys
import time
import pandas as pd
import io
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure backend dir is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services.ai import extract_csv_from_pdf_text

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def evaluate_extraction():
    test_cases = [
        {"pdf": "../test_data/pdf/detailed_technology_saas.pdf", "csv": "../test_data/csv/technology_saas_corp.csv", "name": "Technology"},
        {"pdf": "../test_data/pdf/detailed_manufacturing_heavy.pdf", "csv": "../test_data/csv/manufacturing_industries.csv", "name": "Manufacturing"},
        {"pdf": "../test_data/pdf/detailed_retail_ecommerce.pdf", "csv": "../test_data/csv/retail_ecommerce_inc.csv", "name": "Retail"},
        {"pdf": "../test_data/pdf/detailed_healthcare_biotech.pdf", "csv": "../test_data/csv/healthcare_biotech_ltd.csv", "name": "Healthcare"},
    ]
    
    results = {}
    
    print("Starting LLM Extraction Evaluation Pipeline...")
    
    for case in test_cases:
        pdf_path = os.path.join(os.path.dirname(__file__), case["pdf"])
        csv_path = os.path.join(os.path.dirname(__file__), case["csv"])
        
        print(f"Evaluating {case['name']}...")
        
        # 1. Load Ground Truth
        try:
            gt_df = pd.read_csv(csv_path)
            gt_df.set_index(gt_df.columns[0], inplace=True)
        except Exception as e:
            print(f"  [!] Error loading ground truth for {case['name']}: {e}")
            continue
            
        # 2. Extract from PDF
        try:
            pdf_text = extract_text_from_pdf(pdf_path)
            extracted_csv_str = extract_csv_from_pdf_text(pdf_text)
            ext_df = pd.read_csv(io.StringIO(extracted_csv_str))
            ext_df.set_index(ext_df.columns[0], inplace=True)
        except Exception as e:
            print(f"  [!] Error extracting data for {case['name']}: {e}")
            results[case["name"]] = 0.0
            continue
            
        # 3. Compare (Cell-level accuracy)
        total_cells = 0
        correct_cells = 0
        
        for metric in gt_df.index:
            for col in gt_df.columns:
                total_cells += 1
                try:
                    # Match by approximate metric name
                    ext_metric = None
                    for em in ext_df.index:
                        if em.lower().strip() == metric.lower().strip():
                            ext_metric = em
                            break
                    if not ext_metric:
                        continue
                        
                    # Find matching column
                    ext_col = None
                    for ec in ext_df.columns:
                        if ec.strip() == col.strip():
                            ext_col = ec
                            break
                    if not ext_col:
                        continue
                        
                    gt_val = float(str(gt_df.loc[metric, col]).replace(',', '').replace('$', '').replace('(', '-').replace(')', '').strip())
                    ext_val = float(str(ext_df.loc[ext_metric, ext_col]).replace(',', '').replace('$', '').replace('(', '-').replace(')', '').strip())
                    
                    if abs(gt_val - ext_val) < 0.01: # allow minor float differences
                        correct_cells += 1
                except Exception as e:
                    pass
        
        accuracy = (correct_cells / total_cells) * 100 if total_cells > 0 else 0
        print(f"  -> Accuracy: {accuracy:.2f}% ({correct_cells}/{total_cells} cells)")
        results[case["name"]] = accuracy
        time.sleep(2) # rate limit mitigation
        
    # Plot
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
    plt.title("LLM Tabular Data Extraction Accuracy by Sector")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 105)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=12, xytext=(0, 5), textcoords='offset points')
    
    out_path = os.path.join(os.path.dirname(__file__), "paper_assets", "extraction_accuracy.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {out_path}")

if __name__ == "__main__":
    evaluate_extraction()
