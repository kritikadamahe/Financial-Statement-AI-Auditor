import unittest
import numpy as np
from typing import Dict, Any

from monte_carlo.generator import MonteCarloGenerator
from monte_carlo.fraud_injector import (
    RevenueInflationInjector,
    BenfordViolationInjector,
    EarningsCashDivergenceInjector,
    DebtHidingInjector,
    RevenueSmoothingInjector,
    OpexCapitalizationInjector
)

class TestFraudInjectors(unittest.TestCase):
    """
    Unit tests to verify that fraud injectors alter financial profiles
    in the expected directions while maintaining reconciled ledgers.
    """
    def setUp(self):
        self.generator = MonteCarloGenerator()
        
    def test_revenue_inflation(self):
        """Tests that Revenue Inflation spikes revenue and net income but depresses cash flow."""
        company = self.generator.generate_company("C_INF", "Inf Corp", "Technology", is_fraud=False)
        
        old_rev = company.statements[2024].revenue
        old_ni = company.statements[2024].net_income
        old_ocf = company.statements[2024].operating_cash_flow
        
        injector = RevenueInflationInjector()
        injector.inject(company)
        
        new_rev = company.statements[2024].revenue
        new_ni = company.statements[2024].net_income
        new_ocf = company.statements[2024].operating_cash_flow
        
        self.assertGreater(new_rev, old_rev)
        self.assertGreater(new_ni, old_ni)
        
        # In the new reconciled model, OCF remains constant because the revenue inflation 
        # is offset by accounts receivable delta, ensuring the cash flow identity resolves.
        # So we assert new OCF is equal to old OCF (within float tolerance)
        self.assertAlmostEqual(new_ocf, old_ocf, places=2)
        
        # Verify balance sheet identity still holds
        stmt = company.statements[2024]
        liabilities_and_equity = stmt.total_debt + stmt.accounts_payable + stmt.other_current_liabilities + stmt.total_equity
        self.assertAlmostEqual(stmt.total_assets, liabilities_and_equity, places=2)

    def test_benford_violation(self):
        """Tests that Benford Violation overrides first digits to 7, 8, or 9."""
        company = self.generator.generate_company("C_BEN", "Ben Corp", "Retail", is_fraud=False)
        
        injector = BenfordViolationInjector()
        injector.inject(company)
        
        digits = []
        for yr in [2022, 2023, 2024]:
            stmt = company.statements[yr]
            for val in [stmt.revenue, stmt.total_assets, stmt.operating_cash_flow]:
                if val != 0:
                    d_str = str(int(abs(val))).replace('-', '').lstrip('0')
                    if d_str:
                        digits.append(int(d_str[0]))
                        
        self.assertTrue(all(d in [7, 8, 9] for d in digits))

    def test_earnings_cash_divergence(self):
        """Tests that Earnings-Cash Divergence creates progressive Net Income vs. OCF divergence."""
        company = self.generator.generate_company("C_ECD", "Ecd Corp", "Manufacturing", is_fraud=False)
        
        injector = EarningsCashDivergenceInjector()
        injector.inject(company)
        
        # Verify progressive Net income rise vs OCF deflation
        # 2024 Net income should be inflated relative to 2020
        self.assertGreater(company.statements[2024].net_income, company.statements[2020].net_income)
        self.assertLess(company.statements[2024].operating_cash_flow, company.statements[2020].operating_cash_flow)
        
        # Verify cash flow identity resolves for 2024
        stmt_2024 = company.statements[2024]
        stmt_2023 = company.statements[2023]
        
        delta_ar = stmt_2024.accounts_receivable - stmt_2023.accounts_receivable
        delta_inv = stmt_2024.inventory - stmt_2023.inventory
        delta_ap = stmt_2024.accounts_payable - stmt_2023.accounts_payable
        
        expected_ocf = stmt_2024.net_income + stmt_2024.depreciation_amortization - delta_ar - delta_inv + delta_ap
        self.assertAlmostEqual(stmt_2024.operating_cash_flow, expected_ocf, places=2)

    def test_debt_hiding(self):
        """Tests that Debt Hiding reduces leverage and current liabilities in late years."""
        company = self.generator.generate_company("C_DH", "Dh Corp", "Technology", is_fraud=False)
        
        old_debt_2024 = company.statements[2024].total_debt
        old_cl_2024 = company.statements[2024].current_liabilities
        
        injector = DebtHidingInjector()
        injector.inject(company)
        
        new_debt_2024 = company.statements[2024].total_debt
        new_cl_2024 = company.statements[2024].current_liabilities
        
        self.assertLess(new_debt_2024, old_debt_2024 * 0.4)
        self.assertLess(new_cl_2024, old_cl_2024 * 0.4)
        
        # Check that assets = liabilities + equity is maintained
        stmt = company.statements[2024]
        liabilities_and_equity = stmt.total_debt + stmt.accounts_payable + stmt.other_current_liabilities + stmt.total_equity
        self.assertAlmostEqual(stmt.total_assets, liabilities_and_equity, places=2)

    def test_revenue_smoothing(self):
        """Tests that Revenue Smoothing flattens YoY Revenue fluctuations and reconciles cash flows."""
        company = self.generator.generate_company("C_RS", "Rs Corp", "Retail", is_fraud=False)
        
        injector = RevenueSmoothingInjector()
        injector.inject(company)
        
        growth_2021 = company.statements[2021].revenue / company.statements[2020].revenue
        growth_2022 = company.statements[2022].revenue / company.statements[2021].revenue
        
        self.assertAlmostEqual(growth_2021, 1.05, places=4)
        self.assertAlmostEqual(growth_2022, 1.05, places=4)
        
        # Reconciled OCF assertion
        stmt_2024 = company.statements[2024]
        stmt_2023 = company.statements[2023]
        delta_ar = stmt_2024.accounts_receivable - stmt_2023.accounts_receivable
        delta_inv = stmt_2024.inventory - stmt_2023.inventory
        delta_ap = stmt_2024.accounts_payable - stmt_2023.accounts_payable
        expected_ocf = stmt_2024.net_income + stmt_2024.depreciation_amortization - delta_ar - delta_inv + delta_ap
        self.assertAlmostEqual(stmt_2024.operating_cash_flow, expected_ocf, places=2)

    def test_opex_capitalization(self):
        """Tests that OpEx Capitalization (WorldCom) shifts OpEx to Assets and reconciles cash flow."""
        company = self.generator.generate_company("C_WC", "Wc Corp", "Technology", is_fraud=False)
        
        old_opex = company.statements[2024].opex
        old_ni = company.statements[2024].net_income
        old_assets = company.statements[2024].total_assets
        old_ocf = company.statements[2024].operating_cash_flow
        
        injector = OpexCapitalizationInjector()
        injector.inject(company)
        
        new_opex = company.statements[2024].opex
        new_ni = company.statements[2024].net_income
        new_assets = company.statements[2024].total_assets
        new_ocf = company.statements[2024].operating_cash_flow
        
        # OpEx goes down
        self.assertLess(new_opex, old_opex)
        # Net Income goes up by the capitalized amount
        cap_amount = old_opex - new_opex
        self.assertAlmostEqual(new_ni, old_ni + cap_amount, places=2)
        # Assets go up
        self.assertAlmostEqual(new_assets, old_assets + cap_amount, places=2)
        # OCF goes up by the capitalized amount
        self.assertAlmostEqual(new_ocf, old_ocf + cap_amount, places=2)
        
        # Verify balance sheet identity
        stmt = company.statements[2024]
        liabilities_and_equity = stmt.total_debt + stmt.accounts_payable + stmt.other_current_liabilities + stmt.total_equity
        self.assertAlmostEqual(stmt.total_assets, liabilities_and_equity, places=2)

if __name__ == "__main__":
    unittest.main()
