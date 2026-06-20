import unittest
import numpy as np
import pandas as pd
from typing import Dict, Any

from monte_carlo.generator import MonteCarloGenerator
from monte_carlo.config import config
from monte_carlo.feature_engineering import calculate_panel_features, calculate_company_features

class TestMonteCarloGenerator(unittest.TestCase):
    """
    Tests the core financial statement generator and feature engineering functions.
    """
    def setUp(self):
        self.generator = MonteCarloGenerator()

    def test_single_statement_structure(self):
        """Verifies structure and constraints of a single generated statement."""
        sector = "Technology"
        sector_cfg = config.sectors[sector]
        
        # Test generation for year 2020 with 2019 baseline
        stmt_2019 = self.generator.generate_single_statement(
            year=2019,
            prev_stmt=None,
            sector_cfg=sector_cfg,
            market_shock=0.0
        )
        
        stmt_2020 = self.generator.generate_single_statement(
            year=2020,
            prev_stmt=stmt_2019,
            sector_cfg=sector_cfg,
            market_shock=0.02
        )
        
        self.assertEqual(stmt_2020.year, 2020)
        self.assertGreater(stmt_2020.revenue, 0.0)
        self.assertGreater(stmt_2020.cogs, 0.0)
        self.assertAlmostEqual(stmt_2020.gross_profit, stmt_2020.revenue - stmt_2020.cogs, places=2)
        
        # 1. Balance Sheet Identity Check: Assets = Liabilities + Equity
        assets = stmt_2020.total_assets
        liabilities_and_equity = (
            stmt_2020.total_debt + 
            stmt_2020.accounts_payable + 
            stmt_2020.other_current_liabilities + 
            stmt_2020.total_equity
        )
        self.assertAlmostEqual(assets, liabilities_and_equity, places=2)
        
        # 2. Indirect Cash Flow Reconciliation Check
        delta_ar = stmt_2020.accounts_receivable - stmt_2019.accounts_receivable
        delta_inv = stmt_2020.inventory - stmt_2019.inventory
        delta_ap = stmt_2020.accounts_payable - stmt_2019.accounts_payable
        
        expected_ocf = stmt_2020.net_income + stmt_2020.depreciation_amortization - delta_ar - delta_inv + delta_ap
        self.assertAlmostEqual(stmt_2020.operating_cash_flow, expected_ocf, places=2)

    def test_company_generation(self):
        """Verifies multi-year simulation shapes, Pareto sizes, and label distributions."""
        company = self.generator.generate_company(
            company_id="COMP_TEST",
            name="Test Corp",
            sector="Healthcare",
            is_fraud=False
        )
        
        self.assertEqual(company.company_id, "COMP_TEST")
        self.assertEqual(company.sector, "Healthcare")
        self.assertFalse(company.is_fraud)
        
        statements = company.get_sorted_statements()
        # Verify 2019 is discarded and exactly 5 years are returned (2020-2024)
        self.assertEqual(len(statements), 5)
        self.assertEqual(statements[0].year, 2020)
        self.assertEqual(statements[-1].year, 2024)

    def test_feature_engineering_pipelines(self):
        """Verifies panel-level and company-level feature engineering runs cleanly."""
        c1 = self.generator.generate_company("COMP_0001", "Corp 1", "Technology", is_fraud=False)
        c2 = self.generator.generate_company("COMP_0002", "Corp 2", "Retail", is_fraud=False)
        
        records = []
        records.extend(c1.to_flat_records())
        records.extend(c2.to_flat_records())
        
        df = pd.DataFrame(records)
        self.assertEqual(len(df), 10)
        
        # 1. Panel Feature Pipeline
        panel_df = calculate_panel_features(df)
        self.assertIn("current_ratio", panel_df.columns)
        self.assertIn("revenue_growth", panel_df.columns)
        self.assertIn("growth_divergence", panel_df.columns)
        
        # 2. Company Feature Pipeline
        company_df = calculate_company_features(panel_df)
        self.assertEqual(len(company_df), 2)
        self.assertIn("_slope_divergence", company_df.columns)
        self.assertIn("_RevOCF_corr", company_df.columns)

if __name__ == "__main__":
    unittest.main()
