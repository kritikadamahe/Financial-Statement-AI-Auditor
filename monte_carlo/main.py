import os
import json
import logging
import pandas as pd
from typing import Dict, Any

from .config import config
from .generator import MonteCarloGenerator
from .feature_engineering import calculate_panel_features, calculate_company_features
from .validator import DatasetValidator
from .visualizer import DatasetVisualizer

logger = logging.getLogger("MonteCarloMain")

def generate_methodology_report(validation_summary: Dict[str, Any], output_path: str) -> None:
    """
    Generates a publication-quality methodology report in markdown,
    linking validation outcomes and visual references.
    """
    
    template = r"""# FinAuditAI Monte Carlo Simulation Framework Methodology Report

This report presents the mathematical methodology, implementation details, and validation results of the Monte Carlo simulation framework developed for the **FinAuditAI** research initiative.

---

## 1. Research Context and Synthetic Data Utility
Financial statement fraud is a classic example of a high-impact, low-frequency event. Due to legal constraints, restatements, and settlement confidentiality, public corpora of verified multi-year financial statements detailing specific fraudulent manipulations are sparse. 

To overcome this bottleneck and support supervised/unsupervised anomaly detection training, this framework simulates a realistic corporate population:
* **Normal baseline (1,000 companies)**: Modelled using continuous random growth curves and sector-specific asset/debt profiles.
* **Fraudulent overlay (200 companies)**: Evenly partitioned across 5 distinct, well-documented historical accounting fraud typologies.

By providing a clean, balanced synthetic cohort, machine learning models (such as **Isolation Forest, Local Outlier Factor, One-Class SVM, and PyTorch LSTM Autoencoders**) can learn standard population densities and temporal trajectories to flag anomalies.

---

## 2. Mathematical Simulation Engine
Our engine models annual revenue using **Geometric Brownian Motion (GBM)**, the standard continuous-time stochastic process for asset modeling:

$$S_t = S_{t-1} \exp\left((\mu - \frac{\sigma^2}{2})\Delta t + \sigma \epsilon \sqrt{\Delta t}\right)$$

Where:
* $S_t$ is the current year's revenue.
* $\mu$ represents sector-specific expected annual growth drift (e.g., 12% for Tech, 5% for Manufacturing).
* $\sigma$ represents sector revenue volatility (e.g., 15% for Tech, 8% for Manufacturing).
* $\epsilon \sim N(0, 1)$ represents random macroeconomic/market shocks.

Accounting identities are preserved strictly at each step:
* **Gross Profit**: $\text{Gross Profit}_t = \text{Revenue}_t - \text{COGS}_t$
* **Net Income**: $\text{Net Income}_t = \text{Gross Profit}_t - \text{OpEx}_t$
* **Balance Sheet Identity**: $\text{Total Assets}_t = \text{Total Liabilities}_t + \text{Total Equity}_t$
* **Operating Cash Flow**: Reconciled YoY using the indirect method: $\text{OCF}_t = \text{Net Income}_t + \text{D&A}_t - \Delta\text{AR}_t - \Delta\text{Inventory}_t + \Delta\text{AP}_t$.

---

## 3. Fraud Injection Scenarios
The framework supports six modular, configurable fraud injection strategies:

1. **Revenue Inflation (Enron Typology)**: Inflates revenue and net income in late years (2023-2024) by $2.0\times$ to $3.5\times$. Operating Cash Flow is kept flat or depressed. Balanced on the asset side by inflating Accounts Receivable.
2. **Benford's Law Violation (Outright Fabrication)**: Overwrites the first and first-two digits of all variables with high-digits (7, 8, 9), distorting the natural logarithmic digit distribution.
3. **Earnings-Cash Divergence (Aggressive Accruals)**: Progressively boosts Net Income YoY over the 5-year timespan while dividing Operating Cash Flow by the corresponding factor, reflecting low-quality earnings.
4. **Debt Hiding (Off-Balance-Sheet SPEs)**: Reduces Total Debt and Current Liabilities in 2023-2024 by 70%-95%, reclassifying the liabilities as Equity to maintain sheet balance without dropping assets.
5. **Revenue Smoothing (Channel Stuffing/Cookies)**: Replaces natural revenue growth curves with an artificially stable growth rate (exactly 5% per annum), while Operating Cash Flow maintains natural volatility.
6. **Operating Expense Capitalization (WorldCom Typology)**: Reclassifies 30%-50% of OpEx as CapEx (assets), boosting both Net Income and OCF while lowering reported Operating Expenses.

---

## 4. Dataset Validation and Quality Report

### 4.1 Population Characteristics
* **Total Companies**: __TOTAL_COMPANIES__
* **Normal Companies**: __NORMAL_COMPANIES__
* **Fraudulent Companies**: __FRAUDULENT_COMPANIES__

### 4.2 Accounting Consistency Checks
* **Tolerance Checked**: 5.0%
* **Balance Sheet Violations (excluding Benford)**: __VIOLATIONS__
* **Balance Sheet Violation Rate**: __VIOLATION_RATE__%
* **Cash Flow Reconciliation Violations (excluding Benford)**: __CF_VIOLATIONS__
* **Cash Flow Reconciliation Violation Rate**: __CF_VIOLATION_RATE__%
* **Quality Grade Assigned**: **__QUALITY_GRADE__**

### 4.3 Injected Fraud Verification Signatures
The validation engine verified the fraud signatures against the normal baseline population:
"""

    # Populate aggregate variables
    report_content = template
    report_content = report_content.replace("__TOTAL_COMPANIES__", str(validation_summary["total_companies"]))
    report_content = report_content.replace("__NORMAL_COMPANIES__", str(validation_summary["normal_companies"]))
    report_content = report_content.replace("__FRAUDULENT_COMPANIES__", str(validation_summary["fraudulent_companies"]))
    report_content = report_content.replace("__VIOLATIONS__", str(validation_summary["accounting_violations"]))
    report_content = report_content.replace("__VIOLATION_RATE__", f"{validation_summary['accounting_violation_rate_pct']:.2f}")
    report_content = report_content.replace("__CF_VIOLATIONS__", str(validation_summary["cash_flow_violations"]))
    report_content = report_content.replace("__CF_VIOLATION_RATE__", f"{validation_summary['cash_flow_violation_rate_pct']:.2f}")
    report_content = report_content.replace("__QUALITY_GRADE__", validation_summary["quality_grade"])

    # Append fraud signatures
    for fraud_type, sig in validation_summary["fraud_signatures"].items():
        verified_str = "✓ VERIFIED" if sig["verified"] else "✗ FAILED"
        report_content += f"""
#### {fraud_type.replace('_', ' ').title()}
* Status: **{verified_str}**
* Diagnostic Metric: {sig["metric"]}
* Injected Fraud Value: **{sig["fraud_val"]}** vs. Baseline Population Value: **{sig["normal_val"]}**
"""

    report_content += """
---

## 5. Visual Analysis
The following publication-quality visualizations have been generated:

1. **Label Distribution**: `fraud_distribution.png` (displays breakdown of fraud typologies).
2. **Metric Spreads**: `metric_distributions.png` (histograms showing Revenue vs. OCF spreads for normal and fraud).
3. **Pearson Correlation Heatmap**: `correlation_heatmap.png` (shows relationships between engineered financial ratios).
4. **Growth Density Curves**: `growth_distributions.png` (shows YoY growth dynamics of normal companies vs. revenue inflation).
5. **Signature boxplots**: `fraud_type_comparisons.png` (contrasts engineered ratios across fraud types).

---

## 6. Research Limitations and Future Improvements
* **Sub-sector Granularity**: Currently uses 4 major sectors. Future work will introduce specific sub-industries (e.g. SaaS vs Hardware within Technology).
* **Macroeconomic Waves**: Adding systemic multi-year macroeconomic variables (e.g., inflation rates, recessions) to simulate cyclical industry swings.
* **Integrity of Benford Violation**: Direct digit replacements break ledger reconciliation. Future versions will implement a balanced double-entry transaction generator that produces Benford violations organically.
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    logger.info(f"Saved methodology report to {output_path}")

def main() -> None:
    logger.info("Initializing FinAuditAI Monte Carlo Simulation Framework...")
    
    # 1. Generate Population
    generator = MonteCarloGenerator()
    population = generator.generate_population()
    
    # 2. Extract Panel Records
    panel_records = []
    for comp in population:
        panel_records.extend(comp.to_flat_records())
    panel_df = pd.DataFrame(panel_records)
    
    # 3. Save Raw Dataset
    output_dir = os.path.dirname(os.path.abspath(__file__))
    raw_csv_path = os.path.join(output_dir, "synthetic_dataset.csv")
    panel_df.to_csv(raw_csv_path, index=False)
    logger.info(f"Saved raw panel dataset to {raw_csv_path}")
    
    # 4. Feature Engineering
    logger.info("Running feature engineering pipeline...")
    engineered_panel_df = calculate_panel_features(panel_df)
    engineered_company_df = calculate_company_features(engineered_panel_df)
    
    # 5. Run Validation Suite
    logger.info("Running validator suite...")
    validator = DatasetValidator(engineered_panel_df, engineered_company_df)
    validation_summary = validator.run_full_validation()
    
    # 6. Generate Figures
    logger.info("Running visualizer suite...")
    visualizer = DatasetVisualizer(engineered_panel_df, engineered_company_df, output_dir=output_dir)
    visualizer.generate_all_plots()
    
    # 7. Write Metadata JSON
    metadata = {
        "seed": config.seed,
        "num_normal": config.num_normal,
        "num_fraud": config.num_fraud,
        "years": [config.start_year, config.end_year],
        "metrics_generated": list(panel_df.columns),
        "validation_grade": validation_summary["quality_grade"],
        "accounting_violation_rate": validation_summary["accounting_violation_rate_pct"]
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved metadata parameters to {metadata_path}")
    
    # 8. Write Methodology Report
    report_path = os.path.join(output_dir, "dataset_report.md")
    generate_methodology_report(validation_summary, report_path)
    
    logger.info("FinAuditAI Monte Carlo framework run completed successfully!")

if __name__ == "__main__":
    main()
