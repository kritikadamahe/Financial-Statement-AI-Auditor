import pandas as pd
from typing import List, Dict, Any
import yfinance as yf
import time
import math
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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


def _check_benfords_law(metrics_by_year: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    all_numbers = []
    for year_data in metrics_by_year.values():
        for val in year_data.values():
            if val != 0 and not pd.isna(val):
                all_numbers.append(abs(val))
    
    if len(all_numbers) < 50: # Need sufficient sample size for statistical validity
        return None
        
    first_digits = []
    for num in all_numbers:
        digit_str = str(num).replace('.', '').replace('-', '').lstrip('0')
        if digit_str:
            first_digits.append(int(digit_str[0]))
            
    observed_counts = [first_digits.count(d) for d in range(1, 10)]
    expected_probs = [math.log10(1 + 1/d) for d in range(1, 10)]
    expected_counts = [p * len(first_digits) for p in expected_probs]
    expected_counts = [max(e, 1e-9) for e in expected_counts]
    
    try:
        chi2_stat, p_val = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)
        if p_val < 0.01: # Strict threshold for critical fraud flag
            return {
                "description": f"Benford's Law violation detected (p-value: {p_val:.4f}). The distribution of leading digits deviates significantly from natural expectations, suggesting potential artificial manipulation.",
                "severity": "Critical",
                "related_metrics": ["All Numerical Data"]
            }
    except Exception as e:
        print(f"[ML] Benford test failed: {e}")
        
    return None

def _run_isolation_forest(metrics_by_year: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    df = pd.DataFrame.from_dict(metrics_by_year, orient='index')
    if len(df) < 3:
        return [] 
        
    # Impute missing values with column mean, then 0
    df_imputed = df.fillna(df.mean(numeric_only=True)).fillna(0)
    
    try:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_imputed)
        
        # Isolate exactly the most distinct year if it's statistically distant
        model = IsolationForest(contamination='auto', random_state=42)
        model.fit(scaled_data)
        scores = model.decision_function(scaled_data)
        
        anomalies = []
        for i, score in enumerate(scores):
            # Only flag if it's deeply anomalous (negative score)
            if score < -0.1:
                outlier_year = df.index[i]
                anomalies.append({
                    "description": f"Machine Learning (Isolation Forest) identified fiscal year {outlier_year} as a highly anomalous structural outlier compared to historical baselines.",
                    "severity": "High",
                    "related_metrics": ["Multi-dimensional structural deviation"]
                })
        return anomalies
    except Exception as e:
        print(f"[ML] Isolation forest failed: {e}")
        return []

def detect_anomalies(metrics_by_year: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Detects anomalies using both rule-based heuristics and Unsupervised Machine Learning.
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
    # 1. Benford's Law
    benford_anomaly = _check_benfords_law(metrics_by_year)
    if benford_anomaly:
        anomalies.append(benford_anomaly)
        
    # 2. Isolation Forest
    if_anomalies = _run_isolation_forest(metrics_by_year)
    anomalies.extend(if_anomalies)

    return anomalies
