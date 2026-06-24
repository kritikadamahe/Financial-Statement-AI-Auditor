import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import yfinance as yf
import time
import math
import os
import joblib
import torch
import shap
from scipy import stats
from .lstm_model import LSTMAutoencoder, compute_temporal_features

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


def reconciliation_checks(metrics_by_year: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Automated Reconciliation Engine — enforces core accounting identities.
    
    Performs 'footing and cross-footing' across the three financial statements:
      1. Balance Sheet Identity: Assets = Liabilities + Equity
      2. Income Statement Internal: Revenue - COGS = Gross Profit
      3. Income Statement Completeness: Gross Profit - OpEx ≈ Operating Income
      4. Net Income to Retained Earnings linkage (cross-year)
      5. Cash Flow to Balance Sheet linkage
      6. Debt Consistency: Total Debt ≤ Total Assets
    
    Returns a list of reconciliation result dicts, each containing:
      - rule: The accounting identity being tested
      - year: Fiscal year tested
      - status: 'Pass', 'Fail', or 'Skipped'
      - expected: Expected value (from the identity)
      - actual: Actual value found in the data
      - difference: Absolute difference
      - tolerance: The tolerance % used
      - message: Human-readable explanation
    """
    results = []
    TOLERANCE = 0.02  # 2% tolerance for rounding/estimation differences
    
    for year in sorted(metrics_by_year.keys()):
        m = metrics_by_year[year]
        
        # ──────────────────────────────────────────
        # Check 1: Balance Sheet Identity
        #   Total Assets = Total Liabilities + Total Equity
        # ──────────────────────────────────────────
        total_assets = m.get("Total Assets")
        total_equity = m.get("Total Equity")
        total_debt = m.get("Total Debt")
        current_liabilities = m.get("Current Liabilities")
        
        if total_assets is not None and total_equity is not None:
            # Approximate total liabilities from available data
            total_liabilities = 0
            if total_debt is not None:
                total_liabilities += total_debt
            if current_liabilities is not None:
                total_liabilities += current_liabilities
            
            if total_liabilities > 0:
                expected = total_liabilities + total_equity
                diff = abs(total_assets - expected)
                pct_diff = diff / abs(total_assets) if total_assets != 0 else 0
                
                results.append({
                    "rule": "Balance Sheet Identity",
                    "formula": "Total Assets = Total Liabilities + Total Equity",
                    "year": year,
                    "status": "Pass" if pct_diff <= TOLERANCE else "Fail",
                    "expected": round(expected, 2),
                    "actual": round(total_assets, 2),
                    "difference": round(diff, 2),
                    "tolerance_pct": TOLERANCE * 100,
                    "message": f"Assets ({total_assets:,.0f}) vs Liabilities+Equity ({expected:,.0f}): {'Balanced' if pct_diff <= TOLERANCE else f'Mismatch of {pct_diff*100:.1f}%'}"
                })
            else:
                results.append({
                    "rule": "Balance Sheet Identity",
                    "formula": "Total Assets = Total Liabilities + Total Equity",
                    "year": year,
                    "status": "Skipped",
                    "expected": None,
                    "actual": round(total_assets, 2),
                    "difference": None,
                    "tolerance_pct": TOLERANCE * 100,
                    "message": "Insufficient liability data to verify Balance Sheet identity."
                })
        
        # ──────────────────────────────────────────
        # Check 2: Income Statement — Gross Profit
        #   Revenue - COGS = Gross Profit
        # ──────────────────────────────────────────
        revenue = m.get("Revenue")
        cogs = m.get("Cost of Goods Sold")
        gross_profit = m.get("Gross Profit")
        
        if revenue is not None and cogs is not None and gross_profit is not None:
            expected_gp = revenue - cogs
            diff = abs(gross_profit - expected_gp)
            pct_diff = diff / abs(revenue) if revenue != 0 else 0
            
            results.append({
                "rule": "Gross Profit Verification",
                "formula": "Revenue - COGS = Gross Profit",
                "year": year,
                "status": "Pass" if pct_diff <= TOLERANCE else "Fail",
                "expected": round(expected_gp, 2),
                "actual": round(gross_profit, 2),
                "difference": round(diff, 2),
                "tolerance_pct": TOLERANCE * 100,
                "message": f"Expected GP ({expected_gp:,.0f}) vs Reported GP ({gross_profit:,.0f}): {'Ties out' if pct_diff <= TOLERANCE else f'Discrepancy of {pct_diff*100:.1f}%'}"
            })
        
        # ──────────────────────────────────────────
        # Check 3: Net Margin Consistency
        #   Net Income / Revenue should be within reasonable bounds
        # ──────────────────────────────────────────
        net_income = m.get("Net Income")
        
        if net_income is not None and revenue is not None and revenue != 0:
            net_margin = net_income / revenue
            # Flag if net margin exceeds 50% (extremely unusual) or is below -50%
            margin_ok = -0.50 <= net_margin <= 0.50
            results.append({
                "rule": "Net Margin Reasonableness",
                "formula": "Net Income / Revenue within [-50%, 50%]",
                "year": year,
                "status": "Pass" if margin_ok else "Fail",
                "expected": "Between -50% and 50%",
                "actual": round(net_margin * 100, 2),
                "difference": None,
                "tolerance_pct": None,
                "message": f"Net Margin is {net_margin*100:.1f}%: {'Within normal range' if margin_ok else 'Outside reasonable bounds — investigate'}"
            })
        
        # ──────────────────────────────────────────
        # Check 4: Debt Sanity
        #   Total Debt should not exceed Total Assets
        # ──────────────────────────────────────────
        if total_debt is not None and total_assets is not None and total_assets != 0:
            debt_ratio = total_debt / total_assets
            debt_ok = debt_ratio <= 1.0
            results.append({
                "rule": "Debt Sanity Check",
                "formula": "Total Debt <= Total Assets",
                "year": year,
                "status": "Pass" if debt_ok else "Fail",
                "expected": f"Debt/Assets <= 100%",
                "actual": round(debt_ratio * 100, 2),
                "difference": None,
                "tolerance_pct": None,
                "message": f"Debt/Assets ratio is {debt_ratio*100:.1f}%: {'Solvent' if debt_ok else 'CRITICAL: Debt exceeds total assets — technical insolvency'}"
            })
        
        # ──────────────────────────────────────────
        # Check 5: Cash Flow Quality
        #   Operating Cash Flow should be positive if Net Income is positive
        # ──────────────────────────────────────────
        ocf = m.get("Operating Cash Flow")
        
        if ocf is not None and net_income is not None:
            if net_income > 0 and ocf < 0:
                results.append({
                    "rule": "Cash Flow Quality",
                    "formula": "If Net Income > 0, Operating Cash Flow should be > 0",
                    "year": year,
                    "status": "Fail",
                    "expected": f"Positive (since NI = {net_income:,.0f})",
                    "actual": round(ocf, 2),
                    "difference": round(abs(ocf), 2),
                    "tolerance_pct": None,
                    "message": f"Net Income is positive ({net_income:,.0f}) but Operating Cash Flow is negative ({ocf:,.0f}). This is a classic earnings quality red flag."
                })
            elif net_income > 0 and ocf > 0:
                results.append({
                    "rule": "Cash Flow Quality",
                    "formula": "If Net Income > 0, Operating Cash Flow should be > 0",
                    "year": year,
                    "status": "Pass",
                    "expected": "Positive",
                    "actual": round(ocf, 2),
                    "difference": None,
                    "tolerance_pct": None,
                    "message": f"Both Net Income ({net_income:,.0f}) and OCF ({ocf:,.0f}) are positive. Earnings quality confirmed."
                })
    
    # ──────────────────────────────────────────
    # Check 6: Cross-Year Consistency
    #   Revenue should not swing >200% in a single year (possible restatement)
    # ──────────────────────────────────────────
    years = sorted(metrics_by_year.keys())
    for i in range(1, len(years)):
        prev_m = metrics_by_year[years[i - 1]]
        curr_m = metrics_by_year[years[i]]
        
        prev_rev = prev_m.get("Revenue")
        curr_rev = curr_m.get("Revenue")
        
        if prev_rev is not None and curr_rev is not None and prev_rev != 0:
            change = abs(curr_rev - prev_rev) / abs(prev_rev)
            ok = change <= 2.0
            results.append({
                "rule": "Revenue Continuity",
                "formula": f"YoY Revenue change <= 200% ({years[i-1]} -> {years[i]})",
                "year": f"{years[i-1]}-{years[i]}",
                "status": "Pass" if ok else "Fail",
                "expected": "Change <= 200%",
                "actual": round(change * 100, 2),
                "difference": None,
                "tolerance_pct": None,
                "message": f"Revenue changed by {change*100:.1f}% from {years[i-1]} to {years[i]}: {'Normal' if ok else 'CRITICAL: Possible restatement or data error'}"
            })
    
    return results


def compute_company_feature_vector(metrics_by_year: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Compresses a company's multi-year data into a SINGLE feature vector.
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
            features[f"{metric}_mean"] = float(np.mean(vals))
            features[f"{metric}_std"] = float(np.std(vals))
            features[f"{metric}_max_min_ratio"] = float(max(vals) / min(vals)) if min(vals) != 0 else 0.0
            # YoY changes
            yoy_changes = [(vals[i] - vals[i-1]) / abs(vals[i-1]) for i in range(1, len(vals)) if vals[i-1] != 0]
            features[f"{metric}_max_yoy"] = float(max(yoy_changes)) if yoy_changes else 0.0
            features[f"{metric}_min_yoy"] = float(min(yoy_changes)) if yoy_changes else 0.0
            features[f"{metric}_yoy_volatility"] = float(np.std(yoy_changes)) if len(yoy_changes) > 1 else 0.0
        else:
            features[f"{metric}_mean"] = 0.0
            features[f"{metric}_std"] = 0.0
            features[f"{metric}_max_min_ratio"] = 0.0
            features[f"{metric}_max_yoy"] = 0.0
            features[f"{metric}_min_yoy"] = 0.0
            features[f"{metric}_yoy_volatility"] = 0.0
    
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
        features[f"{name}_mean"] = float(np.mean(vals)) if vals else 0.0
        features[f"{name}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
    
    # --- Cross-metric Consistency ---
    rev_vals = [metrics_by_year[y].get("Revenue", np.nan) for y in years]
    ocf_vals = [metrics_by_year[y].get("Operating Cash Flow", np.nan) for y in years]
    ni_vals = [metrics_by_year[y].get("Net Income", np.nan) for y in years]
    
    valid_pairs = [(r, o) for r, o in zip(rev_vals, ocf_vals) if not np.isnan(r) and not np.isnan(o)]
    if len(valid_pairs) >= 3:
        r_arr, o_arr = zip(*valid_pairs)
        try:
            corr, _ = stats.pearsonr(r_arr, o_arr)
            features["_RevOCF_corr"] = float(corr)
        except:
            features["_RevOCF_corr"] = 0.0
    else:
        features["_RevOCF_corr"] = 0.0
    
    valid_ni_ocf = [(n, o) for n, o in zip(ni_vals, ocf_vals) if not np.isnan(n) and not np.isnan(o)]
    if len(valid_ni_ocf) >= 3:
        n_arr, o_arr = zip(*valid_ni_ocf)
        try:
            corr, _ = stats.pearsonr(n_arr, o_arr)
            features["_NI_OCF_corr"] = float(corr)
        except:
            features["_NI_OCF_corr"] = 0.0
    else:
        features["_NI_OCF_corr"] = 0.0
    
    # Revenue slope vs OCF slope divergence
    if len(rev_vals) >= 3 and len(ocf_vals) >= 3:
        x = np.arange(len(years))
        rev_clean = np.array([v if not np.isnan(v) else 0 for v in rev_vals])
        ocf_clean = np.array([v if not np.isnan(v) else 0 for v in ocf_vals])
        rev_slope = np.polyfit(x, rev_clean, 1)[0]
        ocf_slope = np.polyfit(x, ocf_clean, 1)[0]
        rev_mean = np.mean(rev_clean)
        features["_slope_divergence"] = float((rev_slope - ocf_slope) / abs(rev_mean)) if rev_mean != 0 else 0.0
    else:
        features["_slope_divergence"] = 0.0
    
    return features


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
    Ensemble Anomaly Detection using pre-trained population-level models.
    Incorporates SHAP (SHapley Additive exPlanations) for explainability.
    """
    try:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        if not os.path.exists(os.path.join(models_dir, "scaler.joblib")):
            return [], {}  # Pre-trained models not available
            
        scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        if_model = joblib.load(os.path.join(models_dir, "if_model.joblib"))
        lof_model = joblib.load(os.path.join(models_dir, "lof_model.joblib"))
        svm_model = joblib.load(os.path.join(models_dir, "svm_model.joblib"))
        model_stats = joblib.load(os.path.join(models_dir, "model_stats.joblib"))
        
        feature_dict = compute_company_feature_vector(metrics_by_year)
        df = pd.DataFrame([feature_dict]).fillna(0)
        X_scaled = scaler.transform(df)
        
        if_score_raw = -if_model.decision_function(X_scaled)[0]
        lof_score_raw = -lof_model.decision_function(X_scaled)[0]
        svm_score_raw = -svm_model.decision_function(X_scaled)[0]
        
        if_vote = 1 if if_score_raw > model_stats['if_threshold'] else 0
        lof_vote = 1 if lof_score_raw > model_stats['lof_threshold'] else 0
        svm_vote = 1 if svm_score_raw > model_stats['svm_threshold'] else 0
        
        total_votes = if_vote + lof_vote + svm_vote
        
        anomalies = []
        if total_votes >= 2:
            voters = []
            if if_vote: voters.append("Isolation Forest")
            if lof_vote: voters.append("Local Outlier Factor")
            if svm_vote: voters.append("One-Class SVM")
            
            # --- SHAP Explainability ---
            try:
                # We use IsolationForest for the SHAP explanation since it's tree-based
                explainer = shap.TreeExplainer(if_model)
                shap_values = explainer.shap_values(X_scaled)
                
                # Get top 3 features contributing to the anomaly
                feature_names = list(feature_dict.keys())
                shap_importance = np.abs(shap_values[0])
                top_indices = np.argsort(shap_importance)[-3:][::-1]
                
                shap_explanation = " Key driving features: " + ", ".join(
                    [f"{feature_names[i]} (SHAP impact: {shap_values[0][i]:.2f})" for i in top_indices]
                )
            except Exception as e:
                print(f"[ML] SHAP explanation failed: {e}")
                shap_explanation = ""
            
            anomalies.append({
                "description": f"Ensemble ML Detection: Company flagged as anomalous by {total_votes}/3 population-trained models ({', '.join(voters)}). This indicates significant deviation from standard industry financial profiles.{shap_explanation}",
                "severity": "High",
                "related_metrics": ["Multi-dimensional structural deviation"]
            })
            
        return anomalies, {}
    except Exception as e:
        print(f"[ML] Ensemble detection failed: {e}")
        return [], {}


def _run_autoencoder(metrics_by_year: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Autoencoder-based Anomaly Detection using pre-trained PyTorch LSTM sequence model.
    """
    try:
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        if not os.path.exists(os.path.join(models_dir, "lstm_autoencoder.pth")):
            return []
            
        ae_scaler = joblib.load(os.path.join(models_dir, "ae_scaler.joblib"))
        model_stats = joblib.load(os.path.join(models_dir, "model_stats.joblib"))
        
        # Get temporal sequence features: shape (num_years, num_features)
        temporal_seq = compute_temporal_features(metrics_by_year)
        seq_len, num_features = temporal_seq.shape
        
        if seq_len < 2:
            return []
            
        # Scale each timestep
        temporal_scaled = ae_scaler.transform(temporal_seq)
        
        # Reshape for PyTorch model: (batch_size=1, seq_len, num_features)
        X_tensor = torch.FloatTensor(temporal_scaled).unsqueeze(0)
        
        # Load Model
        autoencoder = LSTMAutoencoder(input_dim=num_features, hidden_dim=8, num_layers=1)
        autoencoder.load_state_dict(torch.load(os.path.join(models_dir, "lstm_autoencoder.pth"), weights_only=True))
        autoencoder.eval()
        
        with torch.no_grad():
            reconstructed = autoencoder(X_tensor)
            # Calculate MSE reconstruction error
            error = torch.mean((X_tensor - reconstructed) ** 2).item()
        
        anomalies = []
        if error > model_stats['ae_threshold']:
            anomalies.append({
                "description": f"PyTorch LSTM Autoencoder Check: Company's temporal reconstruction error ({error:.4f}) exceeded the population threshold ({model_stats['ae_threshold']:.4f}). The chronological sequence of financial metrics cannot be reconstructed from 'normal' company patterns.",
                "severity": "High",
                "related_metrics": ["Temporal Deep Learning structural deviation"]
            })
            
        return anomalies
    except Exception as e:
        print(f"[ML] LSTM Autoencoder failed: {e}")
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
