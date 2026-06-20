import numpy as np
import pandas as pd
from typing import Dict, Any, List
from scipy import stats

def calculate_panel_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes year-level (panel) ratios and growth rates for each company-year.
    Ensures safe division by zero and preserves basic structure.
    """
    df = df.copy()
    
    # Ratios
    df["current_ratio"] = df.apply(
        lambda r: r["current_assets"] / r["current_liabilities"] if r["current_liabilities"] != 0 else 0.0,
        axis=1
    )
    df["debt_to_equity"] = df.apply(
        lambda r: r["total_debt"] / r["total_equity"] if r["total_equity"] != 0 else 0.0,
        axis=1
    )
    df["gross_margin"] = df.apply(
        lambda r: r["gross_profit"] / r["revenue"] if r["revenue"] != 0 else 0.0,
        axis=1
    )
    df["net_profit_margin"] = df.apply(
        lambda r: r["net_income"] / r["revenue"] if r["revenue"] != 0 else 0.0,
        axis=1
    )
    
    # Growth metrics and slopes require grouping by company and sorting by year
    df = df.sort_values(["company_id", "year"])
    
    # YoY Growth
    df["revenue_growth"] = df.groupby("company_id")["revenue"].pct_change().fillna(0.0)
    df["ocf_growth"] = df.groupby("company_id")["operating_cash_flow"].pct_change().fillna(0.0)
    
    # Growth divergence
    df["growth_divergence"] = df["revenue_growth"] - df["ocf_growth"]
    
    return df

def compute_company_feature_vector(metrics_by_year: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Compresses a company's multi-year data into a SINGLE aggregate feature vector.
    This matches exactly the feature set in backend/evaluate_ml.py.
    """
    years = sorted(metrics_by_year.keys())
    features = {}
    
    key_metrics = [
        "Revenue", "Cost of Goods Sold", "Net Income",
        "Current Assets", "Current Liabilities", "Total Assets",
        "Total Debt", "Total Equity", "Operating Cash Flow",
        "Gross Profit", "Operating Expenses"
    ]
    
    for metric in key_metrics:
        vals = [metrics_by_year[y].get(metric, np.nan) for y in years]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) >= 2:
            features[f"{metric}_mean"] = float(np.mean(vals))
            features[f"{metric}_std"] = float(np.std(vals))
            features[f"{metric}_max_min_ratio"] = float(max(vals) / min(vals)) if min(vals) != 0 else 0.0
            
            # YoY changes
            yoy_changes = [
                (vals[i] - vals[i-1]) / abs(vals[i-1]) 
                for i in range(1, len(vals)) 
                if vals[i-1] != 0
            ]
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
            
    # Financial Ratios (averaged across years)
    ratios = {
        "_CurrentRatio": [],
        "_DebtToEquity": [],
        "_NetMargin": [],
        "_GrossMargin": []
    }
    for year in years:
        m = metrics_by_year[year]
        rev = m.get("Revenue", 0.0)
        cogs = m.get("Cost of Goods Sold", 0.0)
        ni = m.get("Net Income", 0.0)
        ca = m.get("Current Assets", 0.0)
        cl = m.get("Current Liabilities", 0.0)
        td = m.get("Total Debt", 0.0)
        te = m.get("Total Equity", 0.0)
        
        if cl != 0: ratios["_CurrentRatio"].append(ca / cl)
        if te != 0: ratios["_DebtToEquity"].append(td / te)
        if rev != 0: ratios["_NetMargin"].append(ni / rev)
        if rev != 0: ratios["_GrossMargin"].append((rev - cogs) / rev)
        
    for name, vals in ratios.items():
        features[f"{name}_mean"] = float(np.mean(vals)) if vals else 0.0
        features[f"{name}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
        
    # Cross-metric consistency
    rev_vals = [metrics_by_year[y].get("Revenue", np.nan) for y in years]
    ocf_vals = [metrics_by_year[y].get("Operating Cash Flow", np.nan) for y in years]
    ni_vals = [metrics_by_year[y].get("Net Income", np.nan) for y in years]
    
    # Revenue vs OCF Correlation
    valid_pairs = [(r, o) for r, o in zip(rev_vals, ocf_vals) if not np.isnan(r) and not np.isnan(o)]
    if len(valid_pairs) >= 3:
        r_arr, o_arr = zip(*valid_pairs)
        try:
            corr, _ = stats.pearsonr(r_arr, o_arr)
            features["_RevOCF_corr"] = float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            features["_RevOCF_corr"] = 0.0
    else:
        features["_RevOCF_corr"] = 0.0
        
    # Net Income vs OCF Correlation
    valid_ni_ocf = [(n, o) for n, o in zip(ni_vals, ocf_vals) if not np.isnan(n) and not np.isnan(o)]
    if len(valid_ni_ocf) >= 3:
        n_arr, o_arr = zip(*valid_ni_ocf)
        try:
            corr, _ = stats.pearsonr(n_arr, o_arr)
            features["_NI_OCF_corr"] = float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            features["_NI_OCF_corr"] = 0.0
    else:
        features["_NI_OCF_corr"] = 0.0
        
    # Revenue slope vs OCF slope divergence
    if len(rev_vals) >= 3 and len(ocf_vals) >= 3:
        x = np.arange(len(years))
        rev_clean = np.array([v if not np.isnan(v) else 0.0 for v in rev_vals])
        ocf_clean = np.array([v if not np.isnan(v) else 0.0 for v in ocf_vals])
        rev_slope = np.polyfit(x, rev_clean, 1)[0]
        ocf_slope = np.polyfit(x, ocf_clean, 1)[0]
        rev_mean = np.mean(rev_clean)
        features["_slope_divergence"] = float((rev_slope - ocf_slope) / abs(rev_mean)) if rev_mean != 0 else 0.0
    else:
        features["_slope_divergence"] = 0.0
        
    return features

def calculate_company_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes company-level aggregate features for a panel dataset.
    Returns a DataFrame with one row per company_id.
    """
    companies = df["company_id"].unique()
    company_records = []
    
    # Mapping df column names to PascalCase required by the aggregate logic
    metric_mapping = {
        "revenue": "Revenue",
        "cogs": "Cost of Goods Sold",
        "net_income": "Net Income",
        "current_assets": "Current Assets",
        "current_liabilities": "Current Liabilities",
        "total_assets": "Total Assets",
        "total_debt": "Total Debt",
        "total_equity": "Total Equity",
        "operating_cash_flow": "Operating Cash Flow",
        "gross_profit": "Gross Profit",
        "opex": "Operating Expenses"
    }
    
    for comp_id in companies:
        comp_df = df[df["company_id"] == comp_id].sort_values("year")
        
        # Build the metrics_by_year dict
        metrics_by_year = {}
        for _, row in comp_df.iterrows():
            year_str = str(int(row["year"]))
            metrics_by_year[year_str] = {
                metric_mapping[col]: float(row[col])
                for col in metric_mapping.keys()
                if col in row
            }
            
        features = compute_company_feature_vector(metrics_by_year)
        features["company_id"] = comp_id
        features["company_name"] = comp_df.iloc[0]["company_name"]
        features["sector"] = comp_df.iloc[0]["sector"]
        features["is_fraud"] = comp_df.iloc[0]["is_fraud"]
        features["fraud_type"] = comp_df.iloc[0]["fraud_type"]
        
        company_records.append(features)
        
    return pd.DataFrame(company_records)
