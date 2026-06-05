import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import yfinance as yf
import time
import math
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Cache to avoid spamming Yahoo Finance APIs
_BENCHMARK_CACHE = {}
_CACHE_TTL = 3600  # 1 hour

PEER_GROUPS = {
    "Technology": ["MSFT", "AAPL", "GOOGL"],
    "Manufacturing": ["CAT", "DE", "MMM"],
    "Retail": ["WMT", "TGT", "COST"],
    "Healthcare": ["JNJ", "PFE", "MRK"]
}

INDUSTRY_BENCHMARKS = {
    "Technology": {
        "Current Ratio": 1.5,
        "Gross Margin (%)": 70.0,
        "Debt-to-Equity Ratio": 0.5,
        "Net Profit Margin (%)": 20.0
    },
    "Manufacturing": {
        "Current Ratio": 1.2,
        "Gross Margin (%)": 35.0,
        "Debt-to-Equity Ratio": 1.0,
        "Net Profit Margin (%)": 10.0
    },
    "Retail": {
        "Current Ratio": 1.1,
        "Gross Margin (%)": 40.0,
        "Debt-to-Equity Ratio": 0.8,
        "Net Profit Margin (%)": 5.0
    },
    "Healthcare": {
        "Current Ratio": 1.4,
        "Gross Margin (%)": 55.0,
        "Debt-to-Equity Ratio": 0.6,
        "Net Profit Margin (%)": 12.0
    }
}

def fetch_realtime_benchmarks(industry: str) -> Dict[str, float]:
    """
    Fetches real-time average metrics for the given industry using yfinance.
    Uses an in-memory cache to prevent slow API calls on repeated requests.
    """
    now = time.time()
    if industry in _BENCHMARK_CACHE:
        cached_data, timestamp = _BENCHMARK_CACHE[industry]
        if now - timestamp < _CACHE_TTL:
            print(f"[Benchmarks] Using cached real-time data for {industry}")
            return cached_data

    tickers = PEER_GROUPS.get(industry, PEER_GROUPS["Technology"])
    metrics = {
        "Current Ratio": [],
        "Gross Margin (%)": [],
        "Debt-to-Equity Ratio": [],
        "Net Profit Margin (%)": []
    }

    print(f"[Benchmarks] Fetching live data for {industry} peers: {tickers}")
    try:
        for ticker_symbol in tickers:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info

            if "currentRatio" in info and info["currentRatio"] is not None:
                metrics["Current Ratio"].append(info["currentRatio"])
            
            if "grossMargins" in info and info["grossMargins"] is not None:
                metrics["Gross Margin (%)"].append(info["grossMargins"] * 100)
            
            if "debtToEquity" in info and info["debtToEquity"] is not None:
                # yfinance debtToEquity is often represented as a percentage (e.g., 50 for 0.5)
                metrics["Debt-to-Equity Ratio"].append(info["debtToEquity"] / 100.0)
            
            if "profitMargins" in info and info["profitMargins"] is not None:
                metrics["Net Profit Margin (%)"].append(info["profitMargins"] * 100)

        # Calculate averages
        avg_metrics = {}
        for metric, values in metrics.items():
            if values:
                avg_metrics[metric] = round(sum(values) / len(values), 2)
            else:
                # Fallback to hardcoded if metric is entirely missing
                avg_metrics[metric] = INDUSTRY_BENCHMARKS.get(industry, INDUSTRY_BENCHMARKS["Technology"])[metric]

        _BENCHMARK_CACHE[industry] = (avg_metrics, now)
        print(f"[Benchmarks] Live averages for {industry}: {avg_metrics}")
        return avg_metrics

    except Exception as e:
        print(f"[Benchmarks] Failed to fetch live data for {industry}: {e}. Falling back to static benchmarks.")
        return INDUSTRY_BENCHMARKS.get(industry, INDUSTRY_BENCHMARKS["Technology"])

def get_industry_benchmarks(industry: str = "Technology") -> Dict[str, float]:
    return fetch_realtime_benchmarks(industry)



def clean_financial_value(raw: str) -> float:
    """
    Converts a financial string value to a float.
    Handles:
      - Parentheses as negatives: (12,500) -> -12500.0
      - Dollar signs: $1,200 -> 1200.0
      - Commas as thousands separators: 1,000,000 -> 1000000.0
      - Plain whitespace
    Raises ValueError if conversion is still not possible.
    """
    cleaned = str(raw).replace(',', '').replace('$', '').replace(' ', '').strip()
    # Parentheses notation for negatives is standard in financial statements
    if cleaned.startswith('(') and cleaned.endswith(')'):
        cleaned = '-' + cleaned[1:-1]
    return float(cleaned)


def extract_financial_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Extracts key metrics by year.
    Expects a DataFrame with 'Metric' as a column (or index) and years as other columns.
    Returns a dictionary structured as { '2023': {'Revenue': 1000, 'Current Assets': 500, ...}, '2024': ... }
    """
    # Standardize common metric names to match our analysis rules
    metric_mapping = {
        "sales": "Revenue",
        "net revenue": "Revenue",
        "total revenue": "Revenue",
        "revenue": "Revenue",
        "cogs": "Cost of Goods Sold",
        "cost of sales": "Cost of Goods Sold",
        "cost of goods sold": "Cost of Goods Sold",
        "operating cashflow": "Operating Cash Flow",
        "cash flow from operations": "Operating Cash Flow",
        "operating cash flow": "Operating Cash Flow",
        "current assets": "Current Assets",
        "current liabilities": "Current Liabilities",
        "total debt": "Total Debt",
        "total equity": "Total Equity",
        "shareholders equity": "Total Equity",
        "total expenses": "Total Expenses",
        "operating expenses": "Total Expenses",
        "net income": "Net Income",
        "profit after tax": "Net Income",
        "total assets": "Total Assets",
        "inventory": "Inventory",
        "accounts receivable": "Accounts Receivable",
    }

    # For MVP, assume the first column contains the metric names
    metric_col = df.columns[0]

    # Robustly detect year columns: cast every header to str first so integer
    # column headers (e.g. Pandas reads "2023" as int 2023) are also matched.
    years = [
        col for col in df.columns
        if str(col).strip().isdigit() or str(col).strip().startswith("20")
    ]

    metrics_by_year = {}
    for year in years:
        year_data = {}
        for _, row in df.iterrows():
            metric_name = str(row[metric_col]).strip()

            # Normalize the metric name using our mapping
            normalized_name = metric_mapping.get(metric_name.lower(), metric_name)

            # Try to convert value to float using the robust cleaner
            try:
                val = clean_financial_value(str(row[year]))
                if not pd.isna(val):
                    year_data[normalized_name] = val
            except (ValueError, TypeError):
                continue
        metrics_by_year[str(year).strip()] = year_data

    return metrics_by_year


def calculate_ratios(metrics_by_year: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Calculates key financial ratios.
    """
    ratios = []

    for year, metrics in metrics_by_year.items():
        # Current Ratio = Current Assets / Current Liabilities
        if "Current Assets" in metrics and "Current Liabilities" in metrics:
            curr_assets = metrics["Current Assets"]
            curr_liabilities = metrics["Current Liabilities"]
            if curr_liabilities != 0:
                val = curr_assets / curr_liabilities
                ratios.append({"name": "Current Ratio", "value": round(val, 2), "year": year})

        # Gross Margin = (Revenue - Cost of Goods Sold) / Revenue
        if "Revenue" in metrics and "Cost of Goods Sold" in metrics:
            revenue = metrics["Revenue"]
            cogs = metrics["Cost of Goods Sold"]
            if revenue != 0:
                val = ((revenue - cogs) / revenue) * 100
                ratios.append({"name": "Gross Margin (%)", "value": round(val, 2), "year": year})

        # Debt-to-Equity = Total Debt / Total Equity
        if "Total Debt" in metrics and "Total Equity" in metrics:
            debt = metrics["Total Debt"]
            equity = metrics["Total Equity"]
            if equity != 0:
                val = debt / equity
                ratios.append({"name": "Debt-to-Equity Ratio", "value": round(val, 2), "year": year})

        # Net Profit Margin = Net Income / Revenue
        if "Net Income" in metrics and "Revenue" in metrics:
            net_income = metrics["Net Income"]
            revenue = metrics["Revenue"]
            if revenue != 0:
                val = (net_income / revenue) * 100
                ratios.append({"name": "Net Profit Margin (%)", "value": round(val, 2), "year": year})

    return ratios


def _compute_advanced_features(metrics_by_year: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Feature Engineering: Computes derived financial features from raw metrics.
    Returns a DataFrame with one row per year, containing:
      - Raw metrics
      - Financial ratios (Current Ratio, Debt-to-Equity, Net Margin, Gross Margin)
      - Year-over-Year percentage changes
      - Cross-metric consistency scores
    """
    years = sorted(metrics_by_year.keys())
    rows = []
    
    for i, year in enumerate(years):
        m = metrics_by_year[year]
        row = dict(m)  # Start with raw metrics
        
        # --- Financial Ratios ---
        rev = m.get("Revenue", 0)
        cogs = m.get("Cost of Goods Sold", 0)
        net_inc = m.get("Net Income", 0)
        curr_a = m.get("Current Assets", 0)
        curr_l = m.get("Current Liabilities", 0)
        total_debt = m.get("Total Debt", 0)
        total_eq = m.get("Total Equity", 0)
        ocf = m.get("Operating Cash Flow", 0)
        
        row["_CurrentRatio"] = curr_a / curr_l if curr_l != 0 else 0
        row["_DebtToEquity"] = total_debt / total_eq if total_eq != 0 else 0
        row["_NetMargin"] = (net_inc / rev * 100) if rev != 0 else 0
        row["_GrossMargin"] = ((rev - cogs) / rev * 100) if rev != 0 else 0
        
        # --- Cross-Metric Consistency Scores ---
        # Revenue vs Cash Flow divergence (Enron red flag)
        row["_RevCashFlowDivergence"] = abs(rev - ocf) / rev if rev != 0 else 0
        # Net Income vs Cash Flow divergence (earnings quality)
        row["_IncCashFlowDivergence"] = abs(net_inc - ocf) / abs(net_inc) if net_inc != 0 else 0
        
        # --- Year-over-Year Changes ---
        if i > 0:
            prev_m = metrics_by_year[years[i - 1]]
            for metric_name in ["Revenue", "Net Income", "Operating Cash Flow", "Total Debt", "Current Assets"]:
                prev_val = prev_m.get(metric_name, 0)
                curr_val = m.get(metric_name, 0)
                if prev_val != 0:
                    row[f"_YoY_{metric_name}"] = (curr_val - prev_val) / abs(prev_val)
                else:
                    row[f"_YoY_{metric_name}"] = 0
        else:
            for metric_name in ["Revenue", "Net Income", "Operating Cash Flow", "Total Debt", "Current Assets"]:
                row[f"_YoY_{metric_name}"] = 0
        
        rows.append(row)
    
    df = pd.DataFrame(rows, index=years)
    return df


def _check_benfords_law(metrics_by_year: Dict[str, Dict[str, float]]) -> Optional[Dict[str, Any]]:
    """
    Advanced Benford's Law analysis:
      1. First-digit Chi-Square test
      2. First-two-digit Chi-Square test (more granular)
      3. Kolmogorov-Smirnov test (better for smaller samples)
    Returns an anomaly dict if a significant violation is detected.
    """
    all_numbers = []
    for year_data in metrics_by_year.values():
        for val in year_data.values():
            if val != 0 and not pd.isna(val):
                all_numbers.append(abs(val))
    
    if len(all_numbers) < 30:  # Relaxed from 50 to work with 5yrs × 10 metrics
        return None
        
    # --- First-Digit Test ---
    first_digits = []
    first_two_digits = []
    for num in all_numbers:
        digit_str = str(num).replace('.', '').replace('-', '').lstrip('0')
        if digit_str and len(digit_str) >= 1:
            first_digits.append(int(digit_str[0]))
        if digit_str and len(digit_str) >= 2:
            first_two_digits.append(int(digit_str[:2]))
            
    violations = 0
    details = []
    
    # Test 1: First-digit Chi-Square
    observed_counts = [first_digits.count(d) for d in range(1, 10)]
    expected_probs = [math.log10(1 + 1/d) for d in range(1, 10)]
    expected_counts = [p * len(first_digits) for p in expected_probs]
    expected_counts = [max(e, 1e-9) for e in expected_counts]
    
    try:
        chi2_stat, chi2_p = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)
        if chi2_p < 0.05:
            violations += 1
            details.append(f"First-digit Chi² p={chi2_p:.4f}")
    except Exception:
        pass
    
    # Test 2: First-two-digit Chi-Square (more sensitive)
    if len(first_two_digits) >= 20:
        observed_ft = [first_two_digits.count(d) for d in range(10, 100)]
        expected_ft_probs = [math.log10(1 + 1/d) for d in range(10, 100)]
        expected_ft = [p * len(first_two_digits) for p in expected_ft_probs]
        expected_ft = [max(e, 1e-9) for e in expected_ft]
        
        try:
            chi2_ft, p_ft = stats.chisquare(f_obs=observed_ft, f_exp=expected_ft)
            if p_ft < 0.05:
                violations += 1
                details.append(f"First-two-digit Chi² p={p_ft:.4f}")
        except Exception:
            pass
    
    # Test 3: Kolmogorov-Smirnov test against Benford CDF
    if len(first_digits) >= 10:
        benford_cdf = np.cumsum([math.log10(1 + 1/d) for d in range(1, 10)])
        observed_cdf = np.cumsum([first_digits.count(d) / len(first_digits) for d in range(1, 10)])
        ks_stat = np.max(np.abs(observed_cdf - benford_cdf))
        # Critical value approximation for KS test
        ks_critical = 1.36 / math.sqrt(len(first_digits))
        if ks_stat > ks_critical:
            violations += 1
            details.append(f"KS statistic={ks_stat:.4f} > critical={ks_critical:.4f}")
    
    if violations >= 2:  # Require at least 2 out of 3 tests to fail
        return {
            "description": f"Benford's Law violation detected via multiple tests ({'; '.join(details)}). The distribution of leading digits deviates significantly from natural expectations, suggesting potential artificial manipulation.",
            "severity": "Critical",
            "related_metrics": ["All Numerical Data"]
        }
        
    return None


def _run_ensemble_detection(metrics_by_year: Dict[str, Dict[str, float]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Ensemble Anomaly Detection combining three unsupervised algorithms:
      1. Isolation Forest (tree-based structural outliers)
      2. Local Outlier Factor (density-based local anomalies)
      3. One-Class SVM (boundary-based normal envelope)
    
    Returns (anomalies_list, scores_dict_by_year).
    Each algorithm votes independently; a year is flagged only if ≥2 algorithms agree.
    """
    feature_df = _compute_advanced_features(metrics_by_year)
    if len(feature_df) < 3:
        return [], {}
        
    # Impute and scale
    df_imputed = feature_df.fillna(feature_df.mean(numeric_only=True)).fillna(0)
    
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_imputed)
        
        years = list(feature_df.index)
        n_samples = len(years)
        
        # --- Model 1: Isolation Forest ---
        if_model = IsolationForest(contamination='auto', random_state=42)
        if_model.fit(scaled_data)
        if_scores = if_model.decision_function(scaled_data)
        if_votes = [1 if s < -0.1 else 0 for s in if_scores]
        
        # --- Model 2: Local Outlier Factor ---
        n_neighbors = min(max(2, n_samples - 1), 20)
        lof_model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto', novelty=False)
        lof_preds = lof_model.fit_predict(scaled_data)
        lof_scores = lof_model.negative_outlier_factor_
        lof_votes = [1 if p == -1 else 0 for p in lof_preds]
        
        # --- Model 3: One-Class SVM ---
        svm_model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.15)
        svm_preds = svm_model.fit_predict(scaled_data)
        svm_scores = svm_model.decision_function(scaled_data)
        svm_votes = [1 if p == -1 else 0 for p in svm_preds]
        
        # --- Ensemble Voting ---
        anomalies = []
        year_scores = {}
        
        for i, year in enumerate(years):
            total_votes = if_votes[i] + lof_votes[i] + svm_votes[i]
            # Normalized ensemble score (0 = definitely normal, 1 = definitely anomalous)
            ensemble_score = (
                (1 - (if_scores[i] + 0.5)) * 0.4 +  # IF contribution (40%)
                (1 - (lof_scores[i] + 1.0)) * 0.3 +  # LOF contribution (30%)
                (1 - (svm_scores[i] + 0.5)) * 0.3     # SVM contribution (30%)
            )
            ensemble_score = max(0.0, min(1.0, ensemble_score))
            year_scores[year] = ensemble_score
            
            # Flag if at least 2 of 3 models agree it's anomalous
            if total_votes >= 2:
                voters = []
                if if_votes[i]: voters.append("Isolation Forest")
                if lof_votes[i]: voters.append("Local Outlier Factor")
                if svm_votes[i]: voters.append("One-Class SVM")
                
                anomalies.append({
                    "description": f"Ensemble ML Detection: Fiscal year {year} flagged as anomalous by {total_votes}/3 models ({', '.join(voters)}). Ensemble anomaly score: {ensemble_score:.2f}.",
                    "severity": "High",
                    "related_metrics": ["Multi-dimensional structural deviation"]
                })
        
        return anomalies, year_scores
        
    except Exception as e:
        print(f"[ML] Ensemble detection failed: {e}")
        return [], {}


def _run_autoencoder(metrics_by_year: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Autoencoder-based Anomaly Detection:
      - Trains a small neural network to reconstruct 'normal' financial profiles.
      - Years with high reconstruction error are flagged as anomalous.
      - Uses sklearn's MLPRegressor as a lightweight autoencoder (no PyTorch needed).
    """
    feature_df = _compute_advanced_features(metrics_by_year)
    if len(feature_df) < 4:
        return []
    
    df_imputed = feature_df.fillna(feature_df.mean(numeric_only=True)).fillna(0)
    
    try:
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df_imputed)
        n_features = scaled_data.shape[1]
        
        # Bottleneck architecture: input -> compressed -> output
        hidden_size = max(2, n_features // 3)
        
        autoencoder = MLPRegressor(
            hidden_layer_sizes=(hidden_size,),
            activation='relu',
            max_iter=500,
            random_state=42,
            tol=1e-5,
            early_stopping=False
        )
        
        # Train autoencoder to reconstruct its own input
        autoencoder.fit(scaled_data, scaled_data)
        
        # Calculate reconstruction error per year
        reconstructed = autoencoder.predict(scaled_data)
        errors = np.mean((scaled_data - reconstructed) ** 2, axis=1)
        
        # Flag years where reconstruction error is > 2 standard deviations above mean
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        threshold = mean_error + 2 * std_error
        
        anomalies = []
        years = list(feature_df.index)
        for i, year in enumerate(years):
            if errors[i] > threshold and std_error > 0.001:  # Avoid flagging when all errors are near-zero
                anomalies.append({
                    "description": f"Autoencoder detected fiscal year {year} as anomalous (reconstruction error: {errors[i]:.4f}, threshold: {threshold:.4f}). The financial profile for this year cannot be reconstructed from the learned 'normal' patterns.",
                    "severity": "High",
                    "related_metrics": ["Deep Learning structural deviation"]
                })
        
        return anomalies
        
    except Exception as e:
        print(f"[ML] Autoencoder failed: {e}")
        return []


def _check_temporal_trajectory(metrics_by_year: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Temporal Trajectory Analysis:
      - Analyzes the slope/trajectory of key metrics over time.
      - Flags divergent trajectories (e.g., revenue trending up but cash flow trending down).
    """
    years = sorted(metrics_by_year.keys())
    if len(years) < 3:
        return []
    
    anomalies = []
    
    # Build time series for key metrics
    metric_series = {}
    for metric_name in ["Revenue", "Net Income", "Operating Cash Flow", "Total Debt"]:
        series = []
        for year in years:
            val = metrics_by_year[year].get(metric_name, None)
            if val is not None:
                series.append(val)
            else:
                series.append(np.nan)
        if not all(np.isnan(v) for v in series if isinstance(v, float)):
            metric_series[metric_name] = series
    
    # Check Revenue vs Operating Cash Flow trajectory divergence
    if "Revenue" in metric_series and "Operating Cash Flow" in metric_series:
        rev = np.array(metric_series["Revenue"])
        ocf = np.array(metric_series["Operating Cash Flow"])
        
        # Calculate slopes using linear regression
        x = np.arange(len(years))
        valid_rev = ~np.isnan(rev)
        valid_ocf = ~np.isnan(ocf)
        
        if sum(valid_rev) >= 3 and sum(valid_ocf) >= 3:
            rev_slope = np.polyfit(x[valid_rev], rev[valid_rev], 1)[0]
            ocf_slope = np.polyfit(x[valid_ocf], ocf[valid_ocf], 1)[0]
            
            # Revenue trending up but cash flow trending down = major red flag
            rev_mean = np.mean(rev[valid_rev])
            if rev_mean != 0:
                rev_trend = rev_slope / abs(rev_mean)
                ocf_trend = ocf_slope / abs(rev_mean)
                
                if rev_trend > 0.05 and ocf_trend < -0.05:
                    anomalies.append({
                        "description": f"Trajectory Divergence: Revenue is trending upward while Operating Cash Flow is trending downward. This pattern is a classic indicator of aggressive revenue recognition or earnings manipulation.",
                        "severity": "Critical",
                        "related_metrics": ["Revenue", "Operating Cash Flow"]
                    })
    
    # Check Net Income vs Operating Cash Flow divergence
    if "Net Income" in metric_series and "Operating Cash Flow" in metric_series:
        ni = np.array(metric_series["Net Income"])
        ocf = np.array(metric_series["Operating Cash Flow"])
        
        valid = ~np.isnan(ni) & ~np.isnan(ocf)
        if sum(valid) >= 3:
            # Compute correlation between Net Income and Cash Flow
            try:
                corr, p_val = stats.pearsonr(ni[valid], ocf[valid])
                if corr < -0.5 and p_val < 0.1:
                    anomalies.append({
                        "description": f"Net Income and Operating Cash Flow show a strong negative correlation (r={corr:.2f}, p={p_val:.4f}). Healthy companies typically show a positive relationship between earnings and cash generation.",
                        "severity": "High",
                        "related_metrics": ["Net Income", "Operating Cash Flow"]
                    })
            except Exception:
                pass
    
    return anomalies


def detect_anomalies(metrics_by_year: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Detects anomalies using a multi-layered approach:
      Layer 1: Rule-based heuristics (YoY thresholds)
      Layer 2: Advanced Benford's Law (3 statistical tests)
      Layer 3: Ensemble ML (Isolation Forest + LOF + One-Class SVM)
      Layer 4: Autoencoder (deep learning reconstruction error)
      Layer 5: Temporal trajectory analysis (slope divergence)
    """
    anomalies = []

    years = sorted(metrics_by_year.keys())
    if len(years) < 2:
        return anomalies  # Need at least 2 years for YoY comparison

    for i in range(1, len(years)):
        prev_year = years[i - 1]
        curr_year = years[i]

        prev_metrics = metrics_by_year[prev_year]
        curr_metrics = metrics_by_year[curr_year]

        # Rule 1: Revenue spiked but operating cash flow dropped
        if "Revenue" in prev_metrics and "Revenue" in curr_metrics and \
           "Operating Cash Flow" in prev_metrics and "Operating Cash Flow" in curr_metrics:

            rev_growth = (curr_metrics["Revenue"] - prev_metrics["Revenue"]) / prev_metrics["Revenue"] if prev_metrics["Revenue"] != 0 else 0
            ocf_growth = (curr_metrics["Operating Cash Flow"] - prev_metrics["Operating Cash Flow"]) / prev_metrics["Operating Cash Flow"] if prev_metrics["Operating Cash Flow"] != 0 else 0

            if rev_growth > 0.10 and ocf_growth < 0:
                anomalies.append({
                    "description": f"Revenue grew by {rev_growth*100:.1f}% but Operating Cash Flow decreased by {abs(ocf_growth)*100:.1f}%.",
                    "severity": "High",
                    "related_metrics": ["Revenue", "Operating Cash Flow"]
                })

        # Rule 2: Unusually high expense growth without corresponding revenue growth
        if "Total Expenses" in prev_metrics and "Total Expenses" in curr_metrics and \
           "Revenue" in prev_metrics and "Revenue" in curr_metrics:

            exp_growth = (curr_metrics["Total Expenses"] - prev_metrics["Total Expenses"]) / prev_metrics["Total Expenses"] if prev_metrics["Total Expenses"] != 0 else 0
            rev_growth = (curr_metrics["Revenue"] - prev_metrics["Revenue"]) / prev_metrics["Revenue"] if prev_metrics["Revenue"] != 0 else 0

            if exp_growth > 0.15 and rev_growth < 0.10:
                anomalies.append({
                    "description": f"Total Expenses grew significantly ({exp_growth*100:.1f}%) while Revenue growth was low ({rev_growth*100:.1f}%).",
                    "severity": "Medium",
                    "related_metrics": ["Total Expenses", "Revenue"]
                })

        # Rule 3: Generic large variance detector (>40% swing in any metric)
        for metric, prev_val in prev_metrics.items():
            if metric in curr_metrics and prev_val != 0:
                curr_val = curr_metrics[metric]
                growth = (curr_val - prev_val) / abs(prev_val)
                if growth > 0.40:
                    anomalies.append({
                        "description": f"Significant spike in {metric}: increased by {growth*100:.1f}% year-over-year.",
                        "severity": "Medium",
                        "related_metrics": [metric]
                    })
                elif growth < -0.40:
                    anomalies.append({
                        "description": f"Significant drop in {metric}: decreased by {abs(growth)*100:.1f}% year-over-year.",
                        "severity": "High",
                        "related_metrics": [metric]
                    })

    # --- Machine Learning Overlay ---
    # Layer 2: Advanced Benford's Law (3 tests)
    benford_anomaly = _check_benfords_law(metrics_by_year)
    if benford_anomaly:
        anomalies.append(benford_anomaly)
        
    # Layer 3: Ensemble ML Detection (IF + LOF + OCSVM)
    ensemble_anomalies, _ = _run_ensemble_detection(metrics_by_year)
    anomalies.extend(ensemble_anomalies)
    
    # Layer 4: Autoencoder
    ae_anomalies = _run_autoencoder(metrics_by_year)
    anomalies.extend(ae_anomalies)
    
    # Layer 5: Temporal Trajectory Analysis
    traj_anomalies = _check_temporal_trajectory(metrics_by_year)
    anomalies.extend(traj_anomalies)

    return anomalies
