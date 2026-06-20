# FinAuditAI Monte Carlo Simulation Framework Methodology Report

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
* **Total Companies**: 1200
* **Normal Companies**: 1000
* **Fraudulent Companies**: 200

### 4.2 Accounting Consistency Checks
* **Tolerance Checked**: 5.0%
* **Balance Sheet Violations (excluding Benford)**: 0
* **Balance Sheet Violation Rate**: 0.00%
* **Cash Flow Reconciliation Violations (excluding Benford)**: 0
* **Cash Flow Reconciliation Violation Rate**: 0.00%
* **Quality Grade Assigned**: **A**

### 4.3 Injected Fraud Verification Signatures
The validation engine verified the fraud signatures against the normal baseline population:

#### Revenue Inflation
* Status: **✓ VERIFIED**
* Diagnostic Metric: Avg Slope Divergence
* Injected Fraud Value: **0.3542** vs. Baseline Population Value: **0.057**

#### Benford Violation
* Status: **✓ VERIFIED**
* Diagnostic Metric: Fraction of Leading Digits in {7,8,9}
* Injected Fraud Value: **1.0** vs. Baseline Population Value: **0.1558**

#### Earnings Cash Divergence
* Status: **✓ VERIFIED**
* Diagnostic Metric: Avg Net Income - OCF Correlation
* Injected Fraud Value: **0.2256** vs. Baseline Population Value: **0.8803**

#### Debt Hiding
* Status: **✓ VERIFIED**
* Diagnostic Metric: Avg Debt-to-Equity Ratio
* Injected Fraud Value: **2.2264** vs. Baseline Population Value: **76.4493**

#### Revenue Smoothing
* Status: **✓ VERIFIED**
* Diagnostic Metric: Avg Revenue YoY Volatility
* Injected Fraud Value: **0.0** vs. Baseline Population Value: **0.1019**

#### Opex Capitalization
* Status: **✓ VERIFIED**
* Diagnostic Metric: Avg Operating Expenses Mean (USD)
* Injected Fraud Value: **1203674.62** vs. Baseline Population Value: **1475140.63**

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
