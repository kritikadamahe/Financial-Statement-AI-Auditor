import os
import sys
import numpy as np
import pandas as pd
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services.analysis import _check_benfords_law, _check_temporal_trajectory

# ─────────────────────────────────────────────────
#  Feature Engineering: Company-Level Summary
# ─────────────────────────────────────────────────

def compute_company_feature_vector(metrics_by_year):
    """
    Compresses a company's multi-year data into a SINGLE feature vector.
    This is the key insight: instead of 5 rows (years), we get 1 row (company)
    with rich aggregate statistics that ML models can actually learn from.
    """
    years = sorted(metrics_by_year.keys())
    features = {}
    
    key_metrics = ["Revenue", "Cost of Goods Sold", "Net Income",
                   "Current Assets", "Current Liabilities", "Total Assets",
                   "Total Debt", "Total Equity", "Operating Cash Flow",
                   "Gross Profit", "Operating Expenses"]
    
    for metric in key_metrics:
        vals = [metrics_by_year[y].get(metric, np.nan) for y in years]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) >= 2:
            features[f"{metric}_mean"] = np.mean(vals)
            features[f"{metric}_std"] = np.std(vals)
            features[f"{metric}_max_min_ratio"] = max(vals) / min(vals) if min(vals) != 0 else 0
            # YoY changes
            yoy_changes = [(vals[i] - vals[i-1]) / abs(vals[i-1]) for i in range(1, len(vals)) if vals[i-1] != 0]
            features[f"{metric}_max_yoy"] = max(yoy_changes) if yoy_changes else 0
            features[f"{metric}_min_yoy"] = min(yoy_changes) if yoy_changes else 0
            features[f"{metric}_yoy_volatility"] = np.std(yoy_changes) if len(yoy_changes) > 1 else 0
        else:
            features[f"{metric}_mean"] = 0
            features[f"{metric}_std"] = 0
            features[f"{metric}_max_min_ratio"] = 0
            features[f"{metric}_max_yoy"] = 0
            features[f"{metric}_min_yoy"] = 0
            features[f"{metric}_yoy_volatility"] = 0
    
    # --- Financial Ratios (averaged across years) ---
    ratios = {"_CurrentRatio": [], "_DebtToEquity": [], "_NetMargin": [], "_GrossMargin": []}
    for year in years:
        m = metrics_by_year[year]
        rev = m.get("Revenue", 0)
        cogs = m.get("Cost of Goods Sold", 0)
        ni = m.get("Net Income", 0)
        ca = m.get("Current Assets", 0)
        cl = m.get("Current Liabilities", 0)
        td = m.get("Total Debt", 0)
        te = m.get("Total Equity", 0)
        
        if cl != 0: ratios["_CurrentRatio"].append(ca / cl)
        if te != 0: ratios["_DebtToEquity"].append(td / te)
        if rev != 0: ratios["_NetMargin"].append(ni / rev)
        if rev != 0: ratios["_GrossMargin"].append((rev - cogs) / rev)
    
    for name, vals in ratios.items():
        features[f"{name}_mean"] = np.mean(vals) if vals else 0
        features[f"{name}_std"] = np.std(vals) if len(vals) > 1 else 0
    
    # --- Cross-metric Consistency ---
    rev_vals = [metrics_by_year[y].get("Revenue", np.nan) for y in years]
    ocf_vals = [metrics_by_year[y].get("Operating Cash Flow", np.nan) for y in years]
    ni_vals = [metrics_by_year[y].get("Net Income", np.nan) for y in years]
    
    valid_pairs = [(r, o) for r, o in zip(rev_vals, ocf_vals) if not np.isnan(r) and not np.isnan(o)]
    if len(valid_pairs) >= 3:
        r_arr, o_arr = zip(*valid_pairs)
        try:
            from scipy.stats import pearsonr
            corr, _ = pearsonr(r_arr, o_arr)
            features["_RevOCF_corr"] = corr
        except:
            features["_RevOCF_corr"] = 0
    else:
        features["_RevOCF_corr"] = 0
    
    valid_ni_ocf = [(n, o) for n, o in zip(ni_vals, ocf_vals) if not np.isnan(n) and not np.isnan(o)]
    if len(valid_ni_ocf) >= 3:
        n_arr, o_arr = zip(*valid_ni_ocf)
        try:
            from scipy.stats import pearsonr
            corr, _ = pearsonr(n_arr, o_arr)
            features["_NI_OCF_corr"] = corr
        except:
            features["_NI_OCF_corr"] = 0
    else:
        features["_NI_OCF_corr"] = 0
    
    # Revenue slope vs OCF slope divergence
    if len(rev_vals) >= 3 and len(ocf_vals) >= 3:
        x = np.arange(len(years))
        rev_clean = np.array([v if not np.isnan(v) else 0 for v in rev_vals])
        ocf_clean = np.array([v if not np.isnan(v) else 0 for v in ocf_vals])
        rev_slope = np.polyfit(x, rev_clean, 1)[0]
        ocf_slope = np.polyfit(x, ocf_clean, 1)[0]
        rev_mean = np.mean(rev_clean)
        features["_slope_divergence"] = (rev_slope - ocf_slope) / abs(rev_mean) if rev_mean != 0 else 0
    else:
        features["_slope_divergence"] = 0
    
    return features


# ─────────────────────────────────────────────────
#  Data Generators
# ─────────────────────────────────────────────────

def generate_normal_company():
    base_revenue = 10 ** np.random.uniform(4, 8)
    metrics = {}
    years = ['2020', '2021', '2022', '2023', '2024']
    
    prev_rev = base_revenue
    for year in years:
        growth = np.random.uniform(-0.05, 0.15)
        rev = prev_rev * (1 + growth)
        cogs_ratio = np.random.uniform(0.35, 0.65)
        net_margin = np.random.uniform(0.08, 0.22)
        ocf_ratio = np.random.uniform(0.12, 0.25)
        
        metrics[year] = {
            "Revenue": rev,
            "Cost of Goods Sold": rev * cogs_ratio,
            "Gross Profit": rev * (1 - cogs_ratio),
            "Operating Expenses": rev * np.random.uniform(0.15, 0.30),
            "Net Income": rev * net_margin,
            "Current Assets": rev * np.random.uniform(0.5, 0.9),
            "Total Assets": rev * np.random.uniform(1.2, 2.5),
            "Current Liabilities": rev * np.random.uniform(0.15, 0.35),
            "Total Debt": rev * np.random.uniform(0.2, 0.5),
            "Total Equity": rev * np.random.uniform(0.5, 1.2),
            "Operating Cash Flow": rev * ocf_ratio,
        }
        prev_rev = rev
    return metrics


def generate_fraudulent_company():
    metrics = generate_normal_company()
    fraud_type = np.random.choice([
        "revenue_inflation",
        "benford_violation",
        "earnings_cash_divergence",
        "debt_hiding",
        "revenue_smoothing"
    ])
    
    if fraud_type == "revenue_inflation":
        metrics['2024']["Revenue"] *= np.random.uniform(2.0, 3.5)
        metrics['2024']["Net Income"] *= np.random.uniform(2.5, 4.0)
        metrics['2024']["Operating Cash Flow"] *= np.random.uniform(0.1, 0.3)
        
    elif fraud_type == "benford_violation":
        for year in metrics.keys():
            for m in metrics[year].keys():
                base = str(int(abs(metrics[year][m])))
                if len(base) > 1:
                    metrics[year][m] = float(np.random.choice(['7', '8', '9']) + base[1:])
                    
    elif fraud_type == "earnings_cash_divergence":
        for i, year in enumerate(sorted(metrics.keys())):
            factor = 1 + (i * 0.3)
            metrics[year]["Net Income"] *= factor
            metrics[year]["Operating Cash Flow"] *= (1 / factor)
            
    elif fraud_type == "debt_hiding":
        metrics['2023']["Total Debt"] *= 0.3
        metrics['2024']["Total Debt"] *= 0.05
        metrics['2024']["Total Equity"] *= 3.0
        
    elif fraud_type == "revenue_smoothing":
        base = metrics['2020']["Revenue"]
        for i, year in enumerate(sorted(metrics.keys())):
            metrics[year]["Revenue"] = base * (1.05 ** i)
            metrics[year]["Net Income"] = base * 0.15 * (1.05 ** i)
            metrics[year]["Operating Cash Flow"] = base * np.random.uniform(0.05, 0.35)

    return metrics


# ─────────────────────────────────────────────────
#  Evaluation Engine
# ─────────────────────────────────────────────────

def evaluate_ml():
    print("=" * 60)
    print("  FinAuditAI -- Large-Scale ML Evaluation (N=1200)")
    print("=" * 60)
    
    np.random.seed(42)
    
    normal_companies = []
    fraud_companies = []
    
    print("\nGenerating 1,000 Normal companies...")
    for _ in range(1000):
        normal_companies.append(generate_normal_company())
    
    print("Generating 200 Fraudulent companies (5 fraud types)...")
    for _ in range(200):
        fraud_companies.append(generate_fraudulent_company())
    
    all_companies = normal_companies + fraud_companies
    labels = [0] * 1000 + [1] * 200
    
    # ─────────────────────────────────────────────
    #  Step 1: Compute feature vectors for ALL companies
    # ─────────────────────────────────────────────
    print("\nComputing company-level feature vectors...")
    all_features = [compute_company_feature_vector(c) for c in all_companies]
    feature_df = pd.DataFrame(all_features).fillna(0)
    feature_df = feature_df.replace([np.inf, -np.inf], 0)
    
    # Split: train on normal, score all
    X_train = feature_df.iloc[:1000]  # Normal companies
    X_all = feature_df  # All companies
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X_all)
    
    # ─────────────────────────────────────────────
    #  Step 2: Per-company statistical layers (Benford + Trajectory)
    # ─────────────────────────────────────────────
    print("Running per-company statistical layers...")
    benford_scores = []
    trajectory_scores = []
    for c in all_companies:
        benford_scores.append(1.0 if _check_benfords_law(c) else 0.0)
        trajectory_scores.append(1.0 if _check_temporal_trajectory(c) else 0.0)
    
    # ─────────────────────────────────────────────
    #  Step 3: Population-level ML models (trained on normals)
    # ─────────────────────────────────────────────
    print("Training Isolation Forest on 1,000 normal companies...")
    if_model = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
    if_model.fit(X_train_scaled)
    if_scores = -if_model.decision_function(X_all_scaled)  # Higher = more anomalous
    if_scores = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-9)
    
    print("Training Local Outlier Factor...")
    lof_model = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
    lof_model.fit(X_train_scaled)
    lof_scores = -lof_model.decision_function(X_all_scaled)
    lof_scores = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-9)
    
    print("Training One-Class SVM...")
    svm_model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.05)
    svm_model.fit(X_train_scaled)
    svm_scores = -svm_model.decision_function(X_all_scaled)
    svm_scores = (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min() + 1e-9)
    
    print("Training Autoencoder...")
    ae_scaler = MinMaxScaler()
    X_train_ae = ae_scaler.fit_transform(X_train)
    X_all_ae = ae_scaler.transform(X_all)
    
    n_features = X_train_ae.shape[1]
    hidden = max(3, n_features // 4)
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(hidden,),
        activation='relu',
        max_iter=1000,
        random_state=42,
        tol=1e-6,
        early_stopping=True,
        validation_fraction=0.1
    )
    autoencoder.fit(X_train_ae, X_train_ae)
    reconstructed = autoencoder.predict(X_all_ae)
    ae_errors = np.mean((X_all_ae - reconstructed) ** 2, axis=1)
    ae_scores = (ae_errors - ae_errors.min()) / (ae_errors.max() - ae_errors.min() + 1e-9)
    
    # ─────────────────────────────────────────────
    #  Step 4: Weighted Ensemble Scoring
    # ─────────────────────────────────────────────
    print("\nComputing weighted ensemble scores...")
    ensemble_scores = (
        np.array(if_scores) * 0.25 +
        np.array(lof_scores) * 0.20 +
        np.array(svm_scores) * 0.15 +
        np.array(ae_scores) * 0.15 +
        np.array(benford_scores) * 0.15 +
        np.array(trajectory_scores) * 0.10
    )
    
    # Find optimal threshold using training data statistics
    normal_ensemble = ensemble_scores[:1000]
    threshold = np.percentile(normal_ensemble, 95)  # Top 5% of normals are borderline
    
    predictions = [1 if s > threshold else 0 for s in ensemble_scores]
    
    # ─────────────────────────────────────────────
    #  Results
    # ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS -- Multi-Layer ML Anomaly Detection")
    print("=" * 60)
    print(classification_report(labels, predictions, target_names=["Normal", "Fraudulent"], zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(7, 6))
    sns.set_theme(style="white")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Fraudulent"],
                yticklabels=["Normal", "Fraudulent"],
                annot_kws={"size": 16})
    plt.title("Multi-Layer ML Anomaly Detection\nConfusion Matrix (N=1200)", fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    cm_path = os.path.join(os.path.dirname(__file__), "paper_assets", "ml_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved Confusion Matrix -> {cm_path}")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, ensemble_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='#FF6B35', lw=2.5, label=f'FinAuditAI Ensemble (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='#888888', lw=1.5, linestyle='--', label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.15, color='#FF6B35')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve -- Multi-Layer Fraud Detection (N=1200)\nAUC = {roc_auc:.2f}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    roc_path = os.path.join(os.path.dirname(__file__), "paper_assets", "ml_roc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC Curve -> {roc_path}")
    
    # Per-Layer Detection Breakdown
    print("\n" + "-" * 60)
    print("  Per-Layer Detection Breakdown (Fraud Companies Only)")
    print("-" * 60)
    
    fraud_mask = np.array(labels) == 1
    layer_data = {
        "Benford's Law": np.array(benford_scores)[fraud_mask],
        "Isolation Forest": np.array(if_scores)[fraud_mask],
        "Local Outlier Factor": np.array(lof_scores)[fraud_mask],
        "One-Class SVM": np.array(svm_scores)[fraud_mask],
        "Autoencoder": np.array(ae_scores)[fraud_mask],
        "Trajectory": np.array(trajectory_scores)[fraud_mask],
    }
    
    # For ML layers, use their own threshold (top 5% of normal scores)
    layer_thresholds = {
        "Benford's Law": 0.5,
        "Isolation Forest": np.percentile(np.array(if_scores)[:1000], 95),
        "Local Outlier Factor": np.percentile(np.array(lof_scores)[:1000], 95),
        "One-Class SVM": np.percentile(np.array(svm_scores)[:1000], 95),
        "Autoencoder": np.percentile(np.array(ae_scores)[:1000], 95),
        "Trajectory": 0.5,
    }
    
    layer_names = []
    layer_rates = []
    layer_counts_list = []
    
    for name, scores in layer_data.items():
        thresh = layer_thresholds[name]
        detected = np.sum(scores > thresh)
        rate = detected / len(scores) * 100
        layer_names.append(name)
        layer_rates.append(rate)
        layer_counts_list.append(detected)
        print(f"  {name:30s}: {detected:4d}/{len(scores)} ({rate:.1f}%)")
    
    # Per-Layer Bar Chart
    plt.figure(figsize=(10, 5))
    colors = ['#264653', '#2a9d8f', '#e76f51', '#e9c46a', '#f4a261', '#606c38']
    bars = plt.bar(layer_names, layer_rates, color=colors)
    for bar, count, total in zip(bars, layer_counts_list, [200]*6):
        plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                f'{count}/{total}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.ylabel('Detection Rate (%)', fontsize=12)
    plt.title('Per-Layer Fraud Detection Rate (N=200 Fraud Companies)', fontsize=13, fontweight='bold')
    plt.ylim(0, 110)
    plt.xticks(rotation=20, ha='right')
    plt.grid(axis='y', alpha=0.3)
    layer_path = os.path.join(os.path.dirname(__file__), "paper_assets", "ml_layer_breakdown.png")
    plt.savefig(layer_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved Layer Breakdown -> {layer_path}")
    
    # Individual model ROC comparison
    plt.figure(figsize=(8, 6))
    model_scores = {
        "Isolation Forest": if_scores,
        "LOF": lof_scores,
        "One-Class SVM": svm_scores,
        "Autoencoder": ae_scores,
        "Ensemble (All)": ensemble_scores,
    }
    model_colors = ['#264653', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51']
    for (name, scores), color in zip(model_scores.items(), model_colors):
        fpr_m, tpr_m, _ = roc_curve(labels, scores)
        auc_m = auc(fpr_m, tpr_m)
        lw = 3.0 if name == "Ensemble (All)" else 1.5
        plt.plot(fpr_m, tpr_m, color=color, lw=lw, label=f'{name} (AUC={auc_m:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Model Comparison ROC Curves', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    comp_path = os.path.join(os.path.dirname(__file__), "paper_assets", "ml_model_comparison.png")
    plt.savefig(comp_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Model Comparison -> {comp_path}")


if __name__ == "__main__":
    evaluate_ml()
