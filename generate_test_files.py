import os
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Set seed for reproducibility
np.random.seed(123)

def generate_base_metrics(base_rev=1000000, years=[2020, 2021, 2022, 2023, 2024]):
    metrics = []
    rev = base_rev
    for y in years:
        ocf_ratio = np.random.uniform(0.15, 0.25)
        metrics.append({
            "Year": str(y),
            "Revenue": int(rev),
            "Cost of Goods Sold": int(rev * np.random.uniform(0.4, 0.6)),
            "Gross Profit": int(rev * np.random.uniform(0.4, 0.6)),
            "Operating Expenses": int(rev * np.random.uniform(0.2, 0.3)),
            "Net Income": int(rev * np.random.uniform(0.1, 0.2)),
            "Current Assets": int(rev * np.random.uniform(0.6, 0.9)),
            "Total Assets": int(rev * np.random.uniform(1.2, 2.5)),
            "Current Liabilities": int(rev * np.random.uniform(0.15, 0.35)),
            "Total Debt": int(rev * np.random.uniform(0.2, 0.5)),
            "Total Equity": int(rev * np.random.uniform(0.5, 1.2)),
            "Operating Cash Flow": int(rev * ocf_ratio)
        })
        rev *= np.random.uniform(1.05, 1.15) # 5-15% natural growth
    return metrics

def create_csv(filename, data):
    # Transform list of dicts to the required format (Metric, 2020, 2021, ...)
    df = pd.DataFrame(data).set_index("Year").T
    df.index.name = "Metric"
    df.to_csv(filename)
    print(f"Generated {filename}")

def generate_normal_tech():
    data = generate_base_metrics(1000000)
    create_csv("test_data/Normal/csv/01_normal_tech.csv", data)

def generate_fraud_revenue_inflation():
    data = generate_base_metrics(2000000)
    # Inflate revenue massively in 2023 and 2024, but cash flow drops
    data[-2]["Revenue"] = int(data[-2]["Revenue"] * 2.5)
    data[-2]["Net Income"] = int(data[-2]["Net Income"] * 2.8)
    data[-2]["Operating Cash Flow"] = int(data[-2]["Operating Cash Flow"] * 0.2)
    
    data[-1]["Revenue"] = int(data[-1]["Revenue"] * 3.5)
    data[-1]["Net Income"] = int(data[-1]["Net Income"] * 4.0)
    data[-1]["Operating Cash Flow"] = int(data[-1]["Operating Cash Flow"] * 0.1)
    
    create_csv("test_data/Fraud/csv/02_fraud_revenue_inflation.csv", data)

def generate_fraud_debt_hiding():
    data = generate_base_metrics(5000000)
    # Suddenly hide debt and artificially inflate equity
    data[-1]["Total Debt"] = int(data[-1]["Total Debt"] * 0.05)
    data[-1]["Current Liabilities"] = int(data[-1]["Current Liabilities"] * 0.1)
    data[-1]["Total Equity"] = int(data[-1]["Total Equity"] * 3.0)
    
    create_csv("test_data/Fraud/csv/03_fraud_debt_hiding.csv", data)

def generate_pdf_with_nlp_risk():
    """Generates a PDF with evasive MD&A text to trigger the LLM NLP Risk module."""
    filename = "test_data/Fraud/pdf/04_fraud_nlp_evasive_mda.pdf"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("Annual Financial Report - 2024", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Evasive MD&A Text
    mda_text = (
        "<b>Management's Discussion & Analysis</b><br/><br/>"
        "In the fiscal year 2024, the company engaged in several highly complex and unstructured strategic partnerships "
        "with offshore entities managed by close associates of the executive board. Due to the rapid evolution of our business model, "
        "we have proactively shifted our revenue recognition policy to immediately recognize the lifetime value of multi-year contracts upfront. "
        "While this deviates from prior historical practices, management believes this aggressive accounting estimate better reflects our "
        "future optimism. Furthermore, operating cash flow has been constrained due to unforeseen liquidity challenges and extended "
        "receivables from these related-party shell corporations, raising substantial doubt about our short-term going concern status, "
        "though we anticipate mitigating this by issuing additional highly-leveraged debt instruments."
    )
    story.append(Paragraph(mda_text, styles['Normal']))
    story.append(Spacer(1, 24))
    
    # Financial Table
    data = generate_base_metrics(1500000)
    # Add a bit of anomaly to the data as well
    data[-1]["Revenue"] = int(data[-1]["Revenue"] * 2.0)
    data[-1]["Operating Cash Flow"] = int(data[-1]["Operating Cash Flow"] * 0.3)
    
    df = pd.DataFrame(data).set_index("Year").T
    df.index.name = "Metric"
    
    # Convert DF to list of lists for ReportLab Table
    table_data = [["Metric"] + list(df.columns)]
    for index, row in df.iterrows():
        table_data.append([index] + [str(x) for x in row.values])
        
    t = Table(table_data)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(t)
    doc.build(story)
    print(f"Generated {filename}")


if __name__ == "__main__":
    os.makedirs("test_data/Normal/csv", exist_ok=True)
    os.makedirs("test_data/Normal/pdf", exist_ok=True)
    os.makedirs("test_data/Fraud/csv", exist_ok=True)
    os.makedirs("test_data/Fraud/pdf", exist_ok=True)
    generate_normal_tech()
    generate_fraud_revenue_inflation()
    generate_fraud_debt_hiding()
    generate_pdf_with_nlp_risk()
    print("All test files generated successfully.")
