import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger("MonteCarloValidator")

class DatasetValidator:
    """
    Validates simulated accounting integrity, outlier signals,
    and fraud injection characteristics.
    """
    def __init__(self, panel_df: pd.DataFrame, company_df: pd.DataFrame):
        self.panel_df = panel_df.copy()
        self.company_df = company_df.copy()

    def check_accounting_consistency(self, tolerance: float = 0.05) -> Tuple[int, float]:
        """
        Validates the Balance Sheet identity: Assets = Liabilities + Equity.
        (Liabilities = total_debt + accounts_payable + other_current_liabilities).
        Returns the number of violations and the violation rate (%), excluding
        Benford violations, which are expected to violate consistency checks.
        """
        # Filter out Benford violation as it intentionally breaks accounting structures
        normal_data = self.panel_df[self.panel_df["fraud_type"] != "benford_violation"]
        
        assets = normal_data["total_assets"]
        liabilities_and_equity = (
            normal_data["total_debt"] + 
            normal_data["accounts_payable"] + 
            normal_data["other_current_liabilities"] + 
            normal_data["total_equity"]
        )
        
        discrepancy = np.abs(assets - liabilities_and_equity) / (np.abs(assets) + 1e-9)
        violations = int(np.sum(discrepancy > tolerance))
        total_records = len(normal_data)
        
        violation_rate = (violations / total_records * 100.0) if total_records > 0 else 0.0
        return violations, violation_rate

    def check_cash_flow_reconciliation(self, tolerance: float = 0.1) -> Tuple[int, float]:
        """
        Validates the indirect cash flow identity:
          OCF = Net Income + D&A - Delta_AR - Delta_Inventory + Delta_AP.
        Returns the number of violations and the violation rate (%), excluding Benford.
        """
        # Filter out Benford
        normal_data = self.panel_df[self.panel_df["fraud_type"] != "benford_violation"].copy()
        normal_data = normal_data.sort_values(["company_id", "year"])
        
        # Calculate deltas YoY
        normal_data["prev_ar"] = normal_data.groupby("company_id")["accounts_receivable"].shift(1)
        normal_data["prev_inv"] = normal_data.groupby("company_id")["inventory"].shift(1)
        normal_data["prev_ap"] = normal_data.groupby("company_id")["accounts_payable"].shift(1)
        
        # Filter out first year (2020) since prev year (2019) is deleted in final dataset
        valid_data = normal_data.dropna(subset=["prev_ar", "prev_inv", "prev_ap"])
        
        delta_ar = valid_data["accounts_receivable"] - valid_data["prev_ar"]
        delta_inv = valid_data["inventory"] - valid_data["prev_inv"]
        delta_ap = valid_data["accounts_payable"] - valid_data["prev_ap"]
        
        expected_ocf = valid_data["net_income"] + valid_data["depreciation_amortization"] - delta_ar - delta_inv + delta_ap
        discrepancy = np.abs(valid_data["operating_cash_flow"] - expected_ocf)
        
        violations = int(np.sum(discrepancy > tolerance))
        total_records = len(valid_data)
        
        violation_rate = (violations / total_records * 100.0) if total_records > 0 else 0.0
        return violations, violation_rate

    def verify_fraud_signatures(self) -> Dict[str, Dict[str, Any]]:
        """
        Verifies that fraud injectors successfully altered the financial signatures.
        """
        results = {}
        normal = self.company_df[self.company_df["is_fraud"] == 0]
        
        # 1. Revenue Inflation Verification
        inflated = self.company_df[self.company_df["fraud_type"] == "revenue_inflation"]
        if len(inflated) > 0 and len(normal) > 0:
            avg_slope_div_inf = inflated["_slope_divergence"].mean()
            avg_slope_div_norm = normal["_slope_divergence"].mean()
            results["revenue_inflation"] = {
                "verified": avg_slope_div_inf > avg_slope_div_norm,
                "metric": "Avg Slope Divergence",
                "fraud_val": round(avg_slope_div_inf, 4),
                "normal_val": round(avg_slope_div_norm, 4)
            }
            
        # 2. Benford Violation Verification
        benford_data = self.panel_df[self.panel_df["fraud_type"] == "benford_violation"]
        if len(benford_data) > 0:
            first_digits = []
            for _, row in benford_data.iterrows():
                val = abs(row["revenue"])
                d_str = str(val).replace('.', '').replace('-', '').lstrip('0')
                if d_str:
                    first_digits.append(int(d_str[0]))
            
            fraction_high_digits = sum(1 for d in first_digits if d in [7, 8, 9]) / (len(first_digits) + 1e-9)
            results["benford_violation"] = {
                "verified": fraction_high_digits > 0.5,
                "metric": "Fraction of Leading Digits in {7,8,9}",
                "fraud_val": round(fraction_high_digits, 4),
                "normal_val": 0.1558
            }
            
        # 3. Earnings-Cash Divergence Verification
        ecd = self.company_df[self.company_df["fraud_type"] == "earnings_cash_divergence"]
        if len(ecd) > 0:
            avg_ni_ocf_corr = ecd["_NI_OCF_corr"].mean()
            avg_ni_ocf_corr_norm = normal["_NI_OCF_corr"].mean()
            results["earnings_cash_divergence"] = {
                "verified": avg_ni_ocf_corr < 0.4,
                "metric": "Avg Net Income - OCF Correlation",
                "fraud_val": round(avg_ni_ocf_corr, 4),
                "normal_val": round(avg_ni_ocf_corr_norm, 4)
            }
            
        # 4. Debt Hiding Verification
        dh = self.company_df[self.company_df["fraud_type"] == "debt_hiding"]
        if len(dh) > 0:
            avg_de_ratio = dh["_DebtToEquity_mean"].mean()
            avg_de_ratio_norm = normal["_DebtToEquity_mean"].mean()
            results["debt_hiding"] = {
                "verified": avg_de_ratio < avg_de_ratio_norm,
                "metric": "Avg Debt-to-Equity Ratio",
                "fraud_val": round(avg_de_ratio, 4),
                "normal_val": round(avg_de_ratio_norm, 4)
            }
            
        # 5. Revenue Smoothing Verification
        rs = self.company_df[self.company_df["fraud_type"] == "revenue_smoothing"]
        if len(rs) > 0:
            avg_rev_vol = rs["Revenue_yoy_volatility"].mean()
            avg_rev_vol_norm = normal["Revenue_yoy_volatility"].mean()
            results["revenue_smoothing"] = {
                "verified": avg_rev_vol < avg_rev_vol_norm,
                "metric": "Avg Revenue YoY Volatility",
                "fraud_val": round(avg_rev_vol, 4),
                "normal_val": round(avg_rev_vol_norm, 4)
            }
            
        # 6. OpEx Capitalization (WorldCom Typology) Verification
        oc = self.company_df[self.company_df["fraud_type"] == "opex_capitalization"]
        if len(oc) > 0:
            avg_opex_mean = oc["Operating Expenses_mean"].mean()
            avg_opex_mean_norm = normal["Operating Expenses_mean"].mean()
            results["opex_capitalization"] = {
                "verified": avg_opex_mean < avg_opex_mean_norm,
                "metric": "Avg Operating Expenses Mean (USD)",
                "fraud_val": round(avg_opex_mean, 2),
                "normal_val": round(avg_opex_mean_norm, 2)
            }
            
        return results

    def run_full_validation(self) -> Dict[str, Any]:
        """
        Runs complete quality assurance checks and aggregates findings.
        """
        logger.info("Executing comprehensive dataset validation...")
        
        bs_violations, bs_rate = self.check_accounting_consistency()
        cf_violations, cf_rate = self.check_cash_flow_reconciliation()
        fraud_verifications = self.verify_fraud_signatures()
        
        total_companies = len(self.company_df)
        fraud_count = int(self.company_df["is_fraud"].sum())
        normal_count = total_companies - fraud_count
        
        summary = {
            "total_companies": total_companies,
            "normal_companies": normal_count,
            "fraudulent_companies": fraud_count,
            "accounting_violations": bs_violations,
            "accounting_violation_rate_pct": round(bs_rate, 4),
            "cash_flow_violations": cf_violations,
            "cash_flow_violation_rate_pct": round(cf_rate, 4),
            "fraud_signatures": fraud_verifications,
            "quality_grade": "A" if (bs_rate < 1.0 and cf_rate < 1.0) else "B"
        }
        
        logger.info(f"Validation complete. BS violation rate: {bs_rate:.2f}%. OCF violation rate: {cf_rate:.2f}%. Grade: {summary['quality_grade']}")
        return summary
