import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from .company_model import Company, FinancialStatement

class BaseFraudInjector(ABC):
    """
    Abstract base class for all fraud injectors.
    Every injector must modify the financial statements of a Company in place.
    """
    @abstractmethod
    def inject(self, company: Company) -> None:
        pass

class RevenueInflationInjector(BaseFraudInjector):
    """
    Simulates Revenue Inflation (Enron Typology):
    Inflates Revenue and Net Income in 2023-2024 by a multiplier (e.g., 2.0x to 3.5x).
    Operating Cash Flow is kept at its original level (showing divergence).
    
    To satisfy the indirect cash flow identity:
      OCF = Net Income + D&A - Delta_AR - Delta_Inventory + Delta_AP
    we balance the increase in Net Income (Delta_NI) by calculating Accounts Receivable
    recursively to resolve the identity.
    """
    def __init__(self, inflation_factor_min: float = 2.0, inflation_factor_max: float = 3.5):
        self.inflation_factor_min = inflation_factor_min
        self.inflation_factor_max = inflation_factor_max

    def inject(self, company: Company) -> None:
        years = sorted(company.statements.keys())
        if len(years) < 2:
            return
            
        target_years = years[-2:]
        for yr in target_years:
            stmt = company.statements[yr]
            factor = random.uniform(self.inflation_factor_min, self.inflation_factor_max)
            
            stmt.revenue *= factor
            stmt.net_income *= factor
            stmt.gross_profit = stmt.revenue - stmt.cogs
            
        # Reconcile recursively to preserve double-entry and cash flow balance
        for idx in range(1, len(years)):
            yr = years[idx]
            curr = company.statements[yr]
            prev = company.statements[years[idx-1]]
            
            delta_inv = curr.inventory - prev.inventory
            delta_ap = curr.accounts_payable - prev.accounts_payable
            
            new_ar = prev.accounts_receivable + curr.net_income + curr.depreciation_amortization - delta_inv + delta_ap - curr.operating_cash_flow
            ar_diff = new_ar - curr.accounts_receivable
            
            curr.accounts_receivable = new_ar
            curr.current_assets += ar_diff
            curr.total_assets += ar_diff
            
            curr.total_equity = curr.total_assets - (curr.total_debt + curr.accounts_payable + curr.other_current_liabilities)

class BenfordViolationInjector(BaseFraudInjector):
    """
    Simulates Benford's Law Violation (Outright Number Fabrication):
    Overwrites the leading digits of all generated financial metrics in the statements
    to violate Benford's Law. Digits are heavily biased towards 7, 8, and 9.
    
    Note: Digit-level fabrication intentionally breaks mathematical balancing,
    representing audit reconciliations failing in double-entry ledgers.
    """
    def __init__(self, biased_digits: List[str] = None):
        self.biased_digits = biased_digits or ['7', '8', '9']

    def inject(self, company: Company) -> None:
        for year, stmt in company.statements.items():
            for attr in [
                "revenue", "cogs", "gross_profit", "opex", "net_income",
                "current_assets", "total_assets", "current_liabilities",
                "total_debt", "total_equity", "operating_cash_flow",
                "inventory", "accounts_receivable", "depreciation_amortization",
                "accounts_payable", "other_current_liabilities"
            ]:
                val = getattr(stmt, attr)
                if val == 0:
                    continue
                
                sign = -1.0 if val < 0 else 1.0
                val_abs = abs(val)
                
                val_str = f"{int(val_abs)}"
                if len(val_str) > 1:
                    new_digit = random.choice(self.biased_digits)
                    new_val_str = new_digit + val_str[1:]
                    fraction = val_abs - int(val_abs)
                    new_val = (float(new_val_str) + fraction) * sign
                    setattr(stmt, attr, new_val)

class EarningsCashDivergenceInjector(BaseFraudInjector):
    """
    Simulates progressive Earnings-Cash Divergence (Aggressive Accruals):
    Net Income is progressively inflated YoY, while OCF is deflated.
    
    To maintain the indirect cash flow identity:
      OCF = Net Income + D&A - Delta_AR - Delta_Inventory + Delta_AP
    we solve for Accounts Receivable recursively.
    """
    def __init__(self, divergence_step: float = 0.3):
        self.divergence_step = divergence_step

    def inject(self, company: Company) -> None:
        years = sorted(company.statements.keys())
        
        # 1. Modify Net Income, Revenue, OCF, and Inventory
        for idx, yr in enumerate(years):
            stmt = company.statements[yr]
            factor = 1.0 + (idx * self.divergence_step)
            
            old_ni = stmt.net_income
            stmt.net_income *= factor
            ni_diff = stmt.net_income - old_ni
            stmt.revenue += ni_diff
            stmt.gross_profit = stmt.revenue - stmt.cogs
            
            # Deflate OCF
            stmt.operating_cash_flow /= factor
            
            # Inflate inventory (representing booking fake assets/overproducing to hide COGS)
            stmt.inventory *= (1.0 + idx * 0.1)
            
        # 2. Reconcile recursively using accounts_receivable as the balancing plug
        for idx in range(1, len(years)):
            yr = years[idx]
            curr = company.statements[yr]
            prev = company.statements[years[idx-1]]
            
            delta_inv = curr.inventory - prev.inventory
            delta_ap = curr.accounts_payable - prev.accounts_payable
            
            new_ar = prev.accounts_receivable + curr.net_income + curr.depreciation_amortization - delta_inv + delta_ap - curr.operating_cash_flow
            ar_diff = new_ar - curr.accounts_receivable
            
            curr.accounts_receivable = new_ar
            curr.current_assets += ar_diff
            curr.total_assets += ar_diff
            
            curr.total_equity = curr.total_assets - (curr.total_debt + curr.accounts_payable + curr.other_current_liabilities)

class DebtHidingInjector(BaseFraudInjector):
    """
    Simulates Debt Hiding (Off-Balance-Sheet special purpose vehicles):
    Reduces Total Debt and Current Liabilities in 2023-2024 by 70% to 95%.
    
    To maintain balance sheet balance, we increase Total Equity (reclassifying debt as equity)
    keeping Total Assets unchanged. Since debt/equity reclassification doesn't affect 
    working capital accounts, OCF reconciles perfectly.
    """
    def __init__(self, debt_reduction_min: float = 0.70, debt_reduction_max: float = 0.95):
        self.debt_reduction_min = debt_reduction_min
        self.debt_reduction_max = debt_reduction_max

    def inject(self, company: Company) -> None:
        years = sorted(company.statements.keys())
        if len(years) < 2:
            return
            
        target_years = years[-2:]
        for yr in target_years:
            stmt = company.statements[yr]
            reduction_factor = random.uniform(self.debt_reduction_min, self.debt_reduction_max)
            
            old_debt = stmt.total_debt
            old_cl = stmt.current_liabilities
            
            new_debt = old_debt * (1.0 - reduction_factor)
            new_cl = old_cl * (1.0 - reduction_factor)
            
            debt_diff = old_debt - new_debt
            
            stmt.total_debt = new_debt
            stmt.current_liabilities = new_cl
            
            # Reclassify debt to equity
            stmt.total_equity += debt_diff

class RevenueSmoothingInjector(BaseFraudInjector):
    """
    Simulates Revenue Smoothing:
    Revenue and Net Income grow at a steady 5% rate every year with zero noise.
    Operating Cash Flow is left with its natural volatility.
    
    To maintain the indirect cash flow identity:
      OCF = Net Income + D&A - Delta_AR - Delta_Inventory + Delta_AP
    we make Accounts Receivable the balancing plug. AR is calculated to reconcile the 
    smooth Net Income against the volatile OCF.
    """
    def __init__(self, target_growth: float = 0.05):
        self.target_growth = target_growth

    def inject(self, company: Company) -> None:
        years = sorted(company.statements.keys())
        if not years:
            return
            
        base_stmt = company.statements[years[0]]
        base_rev = base_stmt.revenue
        base_ni = base_stmt.net_income
        
        for idx, yr in enumerate(years[1:], start=1):
            stmt = company.statements[yr]
            prev_stmt = company.statements[years[idx-1]]
            
            expected_rev = base_rev * ((1.0 + self.target_growth) ** idx)
            expected_ni = base_ni * ((1.0 + self.target_growth) ** idx)
            
            stmt.revenue = expected_rev
            stmt.net_income = expected_ni
            stmt.gross_profit = stmt.revenue - stmt.cogs
            
            # Recalculate AR to balance the indirect cash flow equation:
            # OCF = NI + D&A - (AR - AR_prev) - Delta_Inventory + Delta_AP
            # => AR = AR_prev + NI + D&A - Delta_Inventory + Delta_AP - OCF
            delta_inv = stmt.inventory - prev_stmt.inventory
            delta_ap = stmt.accounts_payable - prev_stmt.accounts_payable
            
            new_ar = prev_stmt.accounts_receivable + stmt.net_income + stmt.depreciation_amortization - delta_inv + delta_ap - stmt.operating_cash_flow
            
            ar_diff = new_ar - stmt.accounts_receivable
            
            stmt.accounts_receivable = new_ar
            stmt.current_assets += ar_diff
            stmt.total_assets += ar_diff
            
            # Re-balance Assets = Liabilities + Equity
            stmt.total_equity = stmt.total_assets - (stmt.total_debt + stmt.accounts_payable + stmt.other_current_liabilities)

class OpexCapitalizationInjector(BaseFraudInjector):
    """
    Simulates Operating Expense Capitalization (WorldCom Typology):
    Reduces OpEx in late years (2023-2024) by 30% to 50%, reclassifying it as CapEx
    (adding to Non-Current Assets/PP&E).
    
    This raises Net Income. Under the indirect cash flow identity:
      OCF = Net Income + D&A - Delta_AR - Delta_Inventory + Delta_AP
    since working capital accounts are unaffected, OCF correctly increases by the same amount,
    while Investing Cash Flow decreases (CapEx outflow).
    """
    def __init__(self, cap_rate_min: float = 0.30, cap_rate_max: float = 0.50):
        self.cap_rate_min = cap_rate_min
        self.cap_rate_max = cap_rate_max

    def inject(self, company: Company) -> None:
        years = sorted(company.statements.keys())
        if len(years) < 2:
            return
            
        target_years = years[-2:]
        for yr in target_years:
            stmt = company.statements[yr]
            cap_rate = random.uniform(self.cap_rate_min, self.cap_rate_max)
            
            old_opex = stmt.opex
            stmt.opex = old_opex * (1.0 - cap_rate)
            cap_amount = old_opex - stmt.opex
            
            # Net Income increases by the capitalized amount
            stmt.net_income += cap_amount
            
            # Capitalized assets added to PP&E / total_assets
            stmt.total_assets += cap_amount
            stmt.total_equity += cap_amount
            
            # Cash Flow is boosted under the indirect cash flow formula because NI increases
            # and no working capital accounts changed.
            stmt.operating_cash_flow += cap_amount

# Updated Factory mapping including the WorldCom injector
FRAUD_INJECTOR_MAP = {
    "revenue_inflation": RevenueInflationInjector,
    "benford_violation": BenfordViolationInjector,
    "earnings_cash_divergence": EarningsCashDivergenceInjector,
    "debt_hiding": DebtHidingInjector,
    "revenue_smoothing": RevenueSmoothingInjector,
    "opex_capitalization": OpexCapitalizationInjector
}
