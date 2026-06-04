import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services.analysis import _check_benfords_law, _run_isolation_forest

def generate_normal_company():
    # 5 years, 10 metrics = 50 data points (enough for Benford's Law)
    # Generate log-uniform base revenue to naturally comply with Benford's Law
    base_revenue = 10 ** np.random.uniform(4, 8) 
    metrics = {}
    years = ['2020', '2021', '2022', '2023', '2024']
    
    for year in years:
        # Realistic noise per year
        yr_mult = np.random.uniform(0.9, 1.15)
        rev = base_revenue * yr_mult
        metrics[year] = {
            "Revenue": rev,
            "Cost of Goods Sold": rev * np.random.uniform(0.4, 0.6),
            "Gross Profit": rev * np.random.uniform(0.4, 0.6), # Collinear
            "Operating Expenses": rev * np.random.uniform(0.2, 0.3),
            "Net Income": rev * np.random.uniform(0.1, 0.25),
            "Current Assets": rev * np.random.uniform(0.5, 0.8),
            "Total Assets": rev * np.random.uniform(1.2, 2.0),
            "Current Liabilities": rev * np.random.uniform(0.2, 0.4),
            "Total Debt": rev * np.random.uniform(0.3, 0.6),
            "Operating Cash Flow": rev * np.random.uniform(0.15, 0.3)
        }
    return metrics

def generate_fraudulent_company():
    metrics = generate_normal_company()
    fraud_type = np.random.choice(["structural", "benford"])
    
    if fraud_type == "structural":
        # Massive anomaly in 2024 (e.g., hiding debt, fake revenue)
        metrics['2024']["Revenue"] *= 2.5
        metrics['2024']["Net Income"] *= 3.0
        metrics['2024']["Operating Cash Flow"] *= 0.2  # Cash flow didn't scale with fake revenue
        metrics['2024']["Total Debt"] *= 0.1 # Magically paid off debt
    else:
        # Fraudulent numbers (violate Benford by starting with 8 or 9)
        for year in metrics.keys():
            for m in metrics[year].keys():
                base = str(int(metrics[year][m]))
                if len(base) > 1:
                    # Force leading digit to 8 or 9
                    metrics[year][m] = float(np.random.choice(['8', '9']) + base[1:])
                
    return metrics

def evaluate_ml():
    print("Generating Synthetic Dataset (1,000 Normal, 200 Fraudulent)...")
    np.random.seed(42)
    
    dataset = []
    labels = [] # 0 = Normal, 1 = Fraud
    
    for _ in range(1000):
        dataset.append(generate_normal_company())
        labels.append(0)
        
    for _ in range(200):
        dataset.append(generate_fraudulent_company())
        labels.append(1)
        
    print("Running ML Anomaly Detection Pipeline (1,200 Trials)...")
    predictions = []
    scores = []
    
    # Track statistics
    false_positives = 0
    true_positives = 0
    
    for i, metrics in enumerate(dataset):
        benford_anomaly = _check_benfords_law(metrics)
        if_anomalies = _run_isolation_forest(metrics)
        
        is_fraud = 1 if (benford_anomaly or if_anomalies) else 0
        predictions.append(is_fraud)
        
        # Calculate score (probability of fraud)
        anomaly_count = (1 if benford_anomaly else 0) + len(if_anomalies)
        scores.append(min(1.0, anomaly_count * 0.45))
        
        # Console output for progress
        if i % 200 == 0:
            print(f"Processed {i}/1200 companies...")

    print("\n--- Large-Scale ML Evaluation Report (N=1200) ---")
    print(classification_report(labels, predictions, target_names=["Normal", "Fraudulent"]))
    
    # Plot Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Fraudulent"], yticklabels=["Normal", "Fraudulent"])
    plt.title("ML Anomaly Detection Confusion Matrix (N=1200)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(os.path.dirname(__file__), "paper_assets", "ml_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Saved Confusion Matrix to {cm_path}")
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Fraud Detection (N=1200)')
    plt.legend(loc="lower right")
    roc_path = os.path.join(os.path.dirname(__file__), "paper_assets", "ml_roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC Curve to {roc_path}")

if __name__ == "__main__":
    evaluate_ml()
