"""
FinAuditAI — Real-World Empirical Validation (SEC AAER Case Studies)
=====================================================================
Benchmarks the pre-trained ML pipeline against historically documented
financial statement fraud cases and a healthy peer control group.

Fraud Cases:
  1. Enron Corporation (1997-2001)       — Revenue inflation, SPEs, cash flow manipulation
  2. WorldCom Inc. (1999-2001)           — Capitalizing OpEx as CapEx, reserve manipulation
  3. Wirecard AG (2015-2018)             — Fabricated TPA revenue, fictitious escrow accounts
  4. Luckin Coffee (2017-2019)           — Fabricated ~$300M in sales, inflated transactions

Healthy Peers (Control Group):
  5. Microsoft (FY2017-FY2021)           — Stable tech blue-chip
  6. Procter & Gamble (FY2017-FY2021)    — Consistent consumer goods
  7. Johnson & Johnson (FY2017-FY2021)   — Diversified healthcare
  8. Visa Inc. (FY2017-FY2021)           — Payment processing (Wirecard peer)

Data Sources: SEC 10-K filings (EDGAR), company annual reports, academic case studies.
All monetary values are sourced in their original reporting currency (USD/EUR/RMB millions).

Methodology Note:
  Each company's absolute monetary values are normalized to the synthetic training
  scale (~$1M base revenue) before scoring. This preserves all YoY changes, financial
  ratios, correlations, and trajectory patterns while ensuring compatibility with
  the StandardScaler fitted on the synthetic training population of 1,200 companies.
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
import warnings
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services.analysis import _check_benfords_law, _check_temporal_trajectory
from services.lstm_model import LSTMAutoencoder, compute_temporal_features
from evaluate_ml import compute_company_feature_vector


# ─────────────────────────────────────────────────────────────
#  REAL-WORLD FINANCIAL DATA (from SEC filings / annual reports)
#  All values in millions of reporting currency
# ─────────────────────────────────────────────────────────────

import json

REAL_WORLD_COMPANIES = {}
try:
    _data_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_world_data.json")
    with open(_data_file, "r", encoding="utf-8") as f:
        REAL_WORLD_COMPANIES = json.load(f)
except Exception as e:
    print(f"Error loading real-world dataset: {e}")



# ─────────────────────────────────────────────────────────────
#  UTILITY: Scale Normalization
# ─────────────────────────────────────────────────────────────

def normalize_company_data(metrics_by_year, target_base=1e6):
    """
    Normalize all monetary values so the first year's revenue ≈ target_base.
    This preserves ALL ratios, YoY changes, correlations, and trajectory patterns
    while ensuring compatibility with the StandardScaler fitted on the synthetic
    training population (which used base_revenue ~ 10^4 to 10^8).
    """
    years = sorted(metrics_by_year.keys())
    first_rev = None
    for y in years:
        rev = metrics_by_year[y].get("Revenue", 0)
        if rev > 0:
            first_rev = rev
            break
    if first_rev is None or first_rev == 0:
        return metrics_by_year

    scale = target_base / first_rev
    normalized = {}
    for year, metrics in metrics_by_year.items():
        normalized[year] = {k: v * scale for k, v in metrics.items()}
    return normalized


# ─────────────────────────────────────────────────────────────
#  MAIN EVALUATION
# ─────────────────────────────────────────────────────────────

def evaluate_real_world():
    print("=" * 70)
    print("  FinAuditAI — Real-World Empirical Validation (SEC Case Studies)")
    print("=" * 70)

    # ── Step 1: Load Pre-Trained Models ──
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    print(f"\nLoading pre-trained models from {models_dir}...")

    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    if_model = joblib.load(os.path.join(models_dir, "if_model.joblib"))
    lof_model = joblib.load(os.path.join(models_dir, "lof_model.joblib"))
    svm_model = joblib.load(os.path.join(models_dir, "svm_model.joblib"))
    ae_scaler = joblib.load(os.path.join(models_dir, "ae_scaler.joblib"))
    model_stats = joblib.load(os.path.join(models_dir, "model_stats.joblib"))

    autoencoder = LSTMAutoencoder(input_dim=15, hidden_dim=8, num_layers=1)
    autoencoder.load_state_dict(
        torch.load(os.path.join(models_dir, "lstm_autoencoder.pth"), weights_only=True)
    )
    autoencoder.eval()

    print("All models loaded successfully.\n")

    # ── Step 2: Process Each Company ──
    company_names = []
    labels = []
    results = []

    for name, info in REAL_WORLD_COMPANIES.items():
        print(f"{'─' * 60}")
        print(f"  Scoring: {name} ({info['industry']})")
        print(f"  Label: {'FRAUD' if info['label'] == 1 else 'HEALTHY'}  |  Currency: {info['currency']}")
        if info['fraud_type']:
            print(f"  Fraud Type: {info['fraud_type']}")
        print(f"{'─' * 60}")

        raw_data = info["data"]
        data = normalize_company_data(raw_data)
        company_names.append(name)
        labels.append(info["label"])

        # ── Feature Engineering ──
        feature_dict = compute_company_feature_vector(data)
        df = pd.DataFrame([feature_dict]).fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        # Ensure column alignment with training data
        trained_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else None
        if trained_features is not None:
            for col in trained_features:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[trained_features]

        X_scaled = scaler.transform(df)

        # ── Layer 1: Benford's Law ──
        benford_result = _check_benfords_law(data)
        benford_score = 1.0 if benford_result else 0.0

        # ── Layer 2: Temporal Trajectory ──
        traj_result = _check_temporal_trajectory(data)
        trajectory_score = 1.0 if traj_result else 0.0

        # ── Layer 3: Isolation Forest ──
        if_score_raw = -if_model.decision_function(X_scaled)[0]
        if_score = max(0, min(1, (if_score_raw - model_stats['if_min']) /
                              (model_stats['if_max'] - model_stats['if_min'] + 1e-9)))

        # ── Layer 4: Local Outlier Factor ──
        lof_score_raw = -lof_model.decision_function(X_scaled)[0]
        lof_score = max(0, min(1, (lof_score_raw - model_stats['lof_min']) /
                              (model_stats['lof_max'] - model_stats['lof_min'] + 1e-9)))

        # ── Layer 5: One-Class SVM ──
        svm_score_raw = -svm_model.decision_function(X_scaled)[0]
        svm_score = max(0, min(1, (svm_score_raw - model_stats['svm_min']) /
                              (model_stats['svm_max'] - model_stats['svm_min'] + 1e-9)))

        # ── Layer 6: LSTM Autoencoder ──
        temporal_seq = compute_temporal_features(data)
        # Pad to 5 years if fewer (LSTM trained on 5-year sequences)
        if temporal_seq.shape[0] < 5:
            pad_rows = 5 - temporal_seq.shape[0]
            padding = np.tile(temporal_seq[-1:], (pad_rows, 1))
            temporal_seq = np.vstack([temporal_seq, padding])
        elif temporal_seq.shape[0] > 5:
            temporal_seq = temporal_seq[:5]

        temporal_scaled = ae_scaler.transform(temporal_seq)
        X_tensor = torch.FloatTensor(temporal_scaled).unsqueeze(0)
        with torch.no_grad():
            reconstructed = autoencoder(X_tensor)
            ae_error = torch.mean((X_tensor - reconstructed) ** 2).item()
        ae_score = max(0, min(1, (ae_error - model_stats['ae_min']) /
                              (model_stats['ae_max'] - model_stats['ae_min'] + 1e-9)))

        # ── Weighted Ensemble ──
        ensemble_score = (
            if_score * 0.25 +
            lof_score * 0.20 +
            svm_score * 0.15 +
            ae_score * 0.15 +
            benford_score * 0.15 +
            trajectory_score * 0.10
        )

        # ── NLP Risk Score (simulated from documented textual characteristics) ──
        nlp_risk = info["nlp_risk_score"]

        # ── Final Prediction (with NLP Veto logic) ──
        threshold = model_stats.get('ensemble_threshold',
                                     np.percentile([0.1, 0.2, 0.3], 85))
        # Use a reasonable threshold: the 85th percentile of normal ensemble scores
        # was ~0.20-0.25 in our training evaluation
        prediction_threshold = 0.25

        raw_pred = 1 if ensemble_score > prediction_threshold else 0
        # Apply NLP Veto: if NLP says low risk AND ML flags it, override
        final_pred = raw_pred
        if raw_pred == 1 and nlp_risk < 30:
            final_pred = 0  # Vetoed by clean text

        # ── Store Results ──
        result = {
            "name": name,
            "label": info["label"],
            "label_str": "FRAUD" if info["label"] == 1 else "HEALTHY",
            "benford": benford_score,
            "trajectory": trajectory_score,
            "if_score": if_score,
            "lof_score": lof_score,
            "svm_score": svm_score,
            "ae_score": ae_score,
            "ensemble": ensemble_score,
            "nlp_risk": nlp_risk,
            "raw_pred": raw_pred,
            "final_pred": final_pred,
            "correct": final_pred == info["label"],
        }
        results.append(result)

        # Print per-company summary
        print(f"  Benford's Law:       {'TRIGGERED' if benford_score > 0 else 'Clean':>12}")
        print(f"  Trajectory:          {'TRIGGERED' if trajectory_score > 0 else 'Clean':>12}")
        print(f"  Isolation Forest:    {if_score:>12.4f}  (threshold: {model_stats['if_threshold']:.4f})")
        print(f"  LOF:                 {lof_score:>12.4f}  (threshold: {model_stats['lof_threshold']:.4f})")
        print(f"  One-Class SVM:       {svm_score:>12.4f}  (threshold: {model_stats['svm_threshold']:.4f})")
        print(f"  LSTM Autoencoder:    {ae_score:>12.4f}  (error: {ae_error:.6f})")
        print(f"  ── Ensemble Score:   {ensemble_score:>12.4f}")
        print(f"  ── NLP Risk Score:   {nlp_risk:>12}")
        print(f"  ── ML Prediction:    {'ANOMALOUS' if raw_pred else 'NORMAL':>12}")
        print(f"  ── Final (w/ Veto):  {'ANOMALOUS' if final_pred else 'NORMAL':>12}")
        print(f"  ── Actual Label:     {result['label_str']:>12}")
        print(f"  ── Correct:          {'✓' if result['correct'] else '✗':>12}")
        print()

    # ─────────────────────────────────────────────────
    #  RESULTS SUMMARY
    # ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  REAL-WORLD VALIDATION RESULTS")
    print("=" * 70)

    # Classification Report
    y_true = [r["label"] for r in results]
    y_pred = [r["final_pred"] for r in results]
    print(classification_report(y_true, y_pred,
                                target_names=["Healthy", "Fraud"],
                                zero_division=0))

    # Accuracy summary
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"  Overall Accuracy: {correct}/{total} ({correct/total*100:.0f}%)")
    fraud_correct = sum(1 for r in results if r["label"] == 1 and r["correct"])
    fraud_total = sum(1 for r in results if r["label"] == 1)
    healthy_correct = sum(1 for r in results if r["label"] == 0 and r["correct"])
    healthy_total = sum(1 for r in results if r["label"] == 0)
    print(f"  Fraud Detection:  {fraud_correct}/{fraud_total} fraud companies correctly identified")
    print(f"  Healthy Accuracy: {healthy_correct}/{healthy_total} healthy companies correctly cleared")
    print()

    # ─────────────────────────────────────────────────
    #  GENERATE IEEE PAPER CHARTS
    # ─────────────────────────────────────────────────
    assets_dir = os.path.join(os.path.dirname(__file__), "paper_assets")
    os.makedirs(assets_dir, exist_ok=True)

    # ── Chart 1: Per-Company Results Table ──
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('off')

    table_data = [["Company", "True Label", "Benford", "IF", "LOF", "SVM", "LSTM-AE",
                   "Trajectory", "Ensemble", "NLP Risk", "Prediction", "Correct"]]
    cell_colors = [['#2c3e50'] * 12]

    for r in results:
        row = [
            r["name"],
            r["label_str"],
            f"{'⚠' if r['benford'] > 0 else '✓'}",
            f"{r['if_score']:.3f}",
            f"{r['lof_score']:.3f}",
            f"{r['svm_score']:.3f}",
            f"{r['ae_score']:.3f}",
            f"{'⚠' if r['trajectory'] > 0 else '✓'}",
            f"{r['ensemble']:.3f}",
            f"{r['nlp_risk']}",
            "FRAUD" if r["final_pred"] else "NORMAL",
            "✓" if r["correct"] else "✗"
        ]
        table_data.append(row)

        if r["label"] == 1:
            bg = '#ffcccc' if r["correct"] else '#ff6666'
        else:
            bg = '#ccffcc' if r["correct"] else '#ff6666'
        cell_colors.append([bg] * 12)

    table = ax.table(cellText=table_data, cellColours=cell_colors,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.6)

    # Style header row
    for j in range(12):
        table[0, j].set_text_props(color='white', fontweight='bold')
        table[0, j].set_facecolor('#2c3e50')

    plt.title("Real-World Empirical Validation — Per-Company Results\n"
              "(SEC AAER Fraud Cases vs. Healthy Peer Control Group)",
              fontsize=13, fontweight='bold', pad=20)
    table_path = os.path.join(assets_dir, "real_world_results_table.png")
    plt.savefig(table_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved Results Table → {table_path}")

    # ── Chart 2: Detection Heatmap (Layers × Companies) ──
    layer_names = ["Benford's Law", "Isolation Forest", "LOF", "One-Class SVM",
                   "LSTM Autoencoder", "Trajectory"]
    heatmap_data = []
    for r in results:
        heatmap_data.append([
            r["benford"], r["if_score"], r["lof_score"],
            r["svm_score"], r["ae_score"], r["trajectory"]
        ])

    heatmap_df = pd.DataFrame(heatmap_data,
                               index=[r["name"] for r in results],
                               columns=layer_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_theme(style="white")

    # Add a horizontal line to separate fraud from healthy
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, vmin=0, vmax=1,
                cbar_kws={'label': 'Anomaly Score'})

    # Draw separator line between fraud and healthy companies
    num_frauds = sum(1 for r in results if r["label"] == 1)
    ax.axhline(y=num_frauds, color='black', linewidth=3)

    ax.set_title("Per-Layer Detection Heatmap — Real-World Case Studies\n"
                 "(Top: Fraud Companies | Bottom: Healthy Peers)",
                 fontsize=13, fontweight='bold')
    ax.set_ylabel("")
    plt.xticks(rotation=25, ha='right')
    plt.yticks(rotation=0)

    heatmap_path = os.path.join(assets_dir, "real_world_detection_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Detection Heatmap → {heatmap_path}")

    # ── Chart 3: Ensemble Score Comparison Bar Chart ──
    fig, ax = plt.subplots(figsize=(12, 6))

    names = [r["name"] for r in results]
    scores = [r["ensemble"] for r in results]
    colors = ['#e74c3c' if r["label"] == 1 else '#27ae60' for r in results]
    edge_colors = ['#c0392b' if r["label"] == 1 else '#1e8449' for r in results]

    bars = ax.bar(names, scores, color=colors, edgecolor=edge_colors, linewidth=1.5)

    # Add threshold line
    ax.axhline(y=prediction_threshold, color='#2c3e50', linestyle='--', linewidth=2,
               label=f'Detection Threshold ({prediction_threshold})')

    # Add value labels on bars
    for bar, score, r in zip(bars, scores, results):
        label = f"{score:.3f}"
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                label, ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylabel('Ensemble Anomaly Score', fontsize=12)
    ax.set_title('Ensemble Anomaly Scores — Real-World Companies\n'
                 '(Red = Fraud | Green = Healthy)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_ylim(0, max(scores) * 1.25)
    plt.xticks(rotation=20, ha='right')
    plt.grid(axis='y', alpha=0.3)

    bar_path = os.path.join(assets_dir, "real_world_ensemble_scores.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Ensemble Scores Bar Chart → {bar_path}")

    # ── Chart 4: Confusion Matrix ──
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy", "Fraud"],
                yticklabels=["Healthy", "Fraud"],
                annot_kws={"size": 18}, ax=ax)
    ax.set_title(f"Real-World Validation Confusion Matrix\n(N={len(results)} Companies)",
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)

    cm_path = os.path.join(assets_dir, "real_world_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Confusion Matrix → {cm_path}")

    print(f"\n{'=' * 70}")
    print(f"  All charts saved to {assets_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    evaluate_real_world()
