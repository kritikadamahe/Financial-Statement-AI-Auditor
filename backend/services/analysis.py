import pandas as pd
from typing import List, Dict, Any


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


def detect_anomalies(metrics_by_year: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    """
    Detects basic rule-based anomalies (e.g., Revenue up 40% but Cash Flow down).
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

    return anomalies
