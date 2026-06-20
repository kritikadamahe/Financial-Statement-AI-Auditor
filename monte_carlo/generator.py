import os
import random
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from .config import config
from .company_model import Company, FinancialStatement
from .fraud_injector import FRAUD_INJECTOR_MAP

logger = logging.getLogger("MonteCarloGenerator")

class MonteCarloGenerator:
    """
    Orchestrates the generation of simulated companies.
    Applies systematic macroeconomic shock factors, Pareto size distribution,
    DSO/DIO/DPO accounting logic, and indirect cash flow reconciliation.
    """
    def __init__(self):
        self.sectors = list(config.sectors.keys())
        self.years = list(range(config.start_year - 1, config.end_year + 1)) # Start from 2019 for t-1 baseline
        
        # Generate systematic macroeconomic market shocks: year -> shock
        # This aligns company growth rates to economic cycles
        self.market_shocks = self._generate_market_shocks()
        
    def _generate_market_shocks(self) -> Dict[int, float]:
        shocks = {}
        for yr in self.years:
            shocks[yr] = np.random.normal(0.0, 0.02) # std of 2% market shock
        logger.info(f"Systematic macroeconomic shocks generated: {shocks}")
        return shocks

    def generate_single_statement(
        self,
        year: int,
        prev_stmt: FinancialStatement,
        sector_cfg: Dict[str, Any],
        market_shock: float
    ) -> FinancialStatement:
        """
        Generates a balanced FinancialStatement using realistic working capital
        DSO/DIO/DPO metrics, D&A rates, and systematic drift factors.
        """
        # 1. Revenue GBM step
        if prev_stmt is None:
            # First year initialization (Pareto Power-Law distribution)
            alpha = config.pareto_alpha
            x_min = sector_cfg["base_revenue_min"]
            u = random.uniform(0.0, 0.98) # cap at 0.98 to prevent infinite outliers
            rev = x_min * ((1.0 - u) ** (-1.0 / alpha))
        else:
            # GBM: S_t = S_{t-1} * exp( (mu_adj - sigma^2/2) * dt + sigma * epsilon * sqrt(dt) )
            mu = sector_cfg["revenue_drift"] + market_shock
            sigma = sector_cfg["revenue_volatility"]
            eps = np.random.normal(0, 1)
            growth_exponent = (mu - (sigma ** 2) / 2.0) + sigma * eps
            rev = max(50000.0, prev_stmt.revenue * np.exp(growth_exponent))

        # 2. Derive income statement items
        cogs_ratio = max(0.1, min(0.9, np.random.normal(
            sector_cfg["cogs_ratio_mean"],
            sector_cfg["cogs_ratio_std"]
        )))
        cogs = rev * cogs_ratio
        gross_profit = rev - cogs
        
        opex_ratio = max(0.05, min(0.5, np.random.normal(
            sector_cfg["opex_ratio_mean"],
            sector_cfg["opex_ratio_std"]
        )))
        opex = rev * opex_ratio
        net_income = gross_profit - opex
        
        # 3. Derive Balance Sheet items
        asset_turnover = max(0.2, np.random.normal(
            sector_cfg["asset_turnover_mean"],
            sector_cfg["asset_turnover_std"]
        ))
        total_assets = rev / asset_turnover
        
        # Assets decomposition: PP&E vs Current Assets
        ppe_fraction = random.uniform(0.4, 0.6)
        ppe = total_assets * ppe_fraction
        
        # Depreciation & Amortization
        dep_rate = max(0.01, min(0.20, np.random.normal(
            sector_cfg["depreciation_rate_mean"],
            sector_cfg["depreciation_rate_std"]
        )))
        depreciation_amortization = ppe * dep_rate
        
        # Working Capital Accounts (DSO, DIO, DPO)
        dso = max(5.0, np.random.normal(sector_cfg["dso_mean"], sector_cfg["dso_std"]))
        dio = max(5.0, np.random.normal(sector_cfg["dio_mean"], sector_cfg["dio_std"]))
        dpo = max(5.0, np.random.normal(sector_cfg["dpo_mean"], sector_cfg["dpo_std"]))
        
        accounts_receivable = (rev * dso) / 365.0
        inventory = (cogs * dio) / 365.0
        accounts_payable = (cogs * dpo) / 365.0
        
        # Cash is the plug to balance current assets
        cash = max(10000.0, total_assets * 0.1) # 10% base cash
        current_assets = cash + accounts_receivable + inventory
        
        # Re-verify Total Assets
        total_assets = ppe + current_assets
        
        # Liabilities decomposition
        leverage = max(0.1, min(0.95, np.random.normal(
            sector_cfg["leverage_mean"],
            sector_cfg["leverage_std"]
        )))
        total_debt = total_assets * leverage
        
        other_liabilities_ratio = random.uniform(0.05, 0.15)
        other_current_liabilities = total_assets * other_liabilities_ratio
        current_liabilities = accounts_payable + other_current_liabilities
        
        # Balance sheet identity: Total Equity = Total Assets - Total Liabilities
        total_equity = total_assets - total_debt - accounts_payable - other_current_liabilities
        
        # 4. Cash Flow Statement: Reconciled Indirect Method
        if prev_stmt is None:
            # For 2019 baseline, OCF is just approximated as Net Income + D&A
            operating_cash_flow = net_income + depreciation_amortization
        else:
            delta_ar = accounts_receivable - prev_stmt.accounts_receivable
            delta_inv = inventory - prev_stmt.inventory
            delta_ap = accounts_payable - prev_stmt.accounts_payable
            
            # Indirect OCF Identity
            operating_cash_flow = net_income + depreciation_amortization - delta_ar - delta_inv + delta_ap

        return FinancialStatement(
            year=year,
            revenue=rev,
            cogs=cogs,
            gross_profit=gross_profit,
            opex=opex,
            net_income=net_income,
            current_assets=current_assets,
            total_assets=total_assets,
            current_liabilities=current_liabilities,
            total_debt=total_debt,
            total_equity=total_equity,
            operating_cash_flow=operating_cash_flow,
            inventory=inventory,
            accounts_receivable=accounts_receivable,
            depreciation_amortization=depreciation_amortization,
            accounts_payable=accounts_payable,
            other_current_liabilities=other_current_liabilities
        )

    def generate_company(
        self,
        company_id: str,
        name: str,
        sector: str,
        is_fraud: bool,
        fraud_type: str = "None"
    ) -> Company:
        """
        Simulates a company over its multi-year operational sequence,
        and injects fraud if flagged.
        """
        company = Company(
            company_id=company_id,
            name=name,
            sector=sector,
            is_fraud=is_fraud,
            fraud_type=fraud_type
        )
        
        sector_cfg = config.sectors[sector]
        prev_stmt = None
        
        for yr in self.years:
            shock = self.market_shocks[yr]
            stmt = self.generate_single_statement(yr, prev_stmt, sector_cfg, shock)
            company.add_statement(stmt)
            prev_stmt = stmt
            
        # Ingest fraud if flagged
        if is_fraud and fraud_type != "None":
            injector_cls = FRAUD_INJECTOR_MAP.get(fraud_type)
            if injector_cls:
                injector = injector_cls()
                injector.inject(company)
                
        # After fraud injectors modify the variables, we must make sure OCF is updated 
        # to match the indirect method of the modified figures (unless Benford breaks it).
        if fraud_type != "benford_violation":
            years_sorted = sorted(company.statements.keys())
            for idx in range(1, len(years_sorted)):
                yr = years_sorted[idx]
                curr = company.statements[yr]
                prev = company.statements[years_sorted[idx-1]]
                
                # Re-calculate OCF based on modified variables to maintain cash flow identity
                delta_ar = curr.accounts_receivable - prev.accounts_receivable
                delta_inv = curr.inventory - prev.inventory
                delta_ap = curr.accounts_payable - prev.accounts_payable
                
                curr.operating_cash_flow = curr.net_income + curr.depreciation_amortization - delta_ar - delta_inv + delta_ap
                
        # Remove 2019 baseline statement before completing so output dataset strictly has 2020-2024
        if config.start_year - 1 in company.statements:
            del company.statements[config.start_year - 1]
            
        return company

    def generate_population(self) -> List[Company]:
        """Generates the entire population of normal and fraudulent companies."""
        logger.info(f"Generating synthetic population: {config.num_normal} Normal, {config.num_fraud} Fraudulent")
        
        population: List[Company] = []
        comp_idx = 1
        
        # 1. Normal Companies
        for _ in range(config.num_normal):
            comp_id = f"COMP_{comp_idx:04d}"
            name = f"Corporation {comp_idx:04d} L.P."
            sector = random.choice(self.sectors)
            company = self.generate_company(comp_id, name, sector, is_fraud=False)
            population.append(company)
            comp_idx += 1
            
        # 2. Fraudulent Companies (distributed evenly across 6 types now)
        fraud_types = list(FRAUD_INJECTOR_MAP.keys())
        fraud_counts = {ft: 0 for ft in fraud_types}
        
        for idx in range(config.num_fraud):
            comp_id = f"COMP_{comp_idx:04d}"
            name = f"Enterprise {comp_idx:04d} Corp"
            sector = random.choice(self.sectors)
            
            # Select fraud type round-robin style
            fraud_type = fraud_types[idx % len(fraud_types)]
            company = self.generate_company(
                comp_id, name, sector, is_fraud=True, fraud_type=fraud_type
            )
            population.append(company)
            fraud_counts[fraud_type] += 1
            comp_idx += 1
            
        logger.info(f"Population generated. Fraud type counts: {fraud_counts}")
        return population
