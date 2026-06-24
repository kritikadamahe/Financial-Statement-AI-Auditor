from dataclasses import dataclass, asdict
from typing import Dict, Any, List

@dataclass
class FinancialStatement:
    """
    Represents a single year's financial statement metrics.
    Includes the core financial metrics and reconciled ledger accounts.
    """
    year: int
    revenue: float
    cogs: float
    gross_profit: float
    opex: float
    net_income: float
    current_assets: float
    total_assets: float
    current_liabilities: float
    total_debt: float
    total_equity: float
    operating_cash_flow: float
    inventory: float = 0.0
    accounts_receivable: float = 0.0
    
    # Reconciled Balance Sheet Decomposition Accounts (IEEE Review Fixes)
    depreciation_amortization: float = 0.0
    accounts_payable: float = 0.0
    other_current_liabilities: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class Company:
    """
    Represents a simulated company in the Monte Carlo framework.
    Maintains basic company metadata and a collection of annual financial statements.
    """
    def __init__(
        self,
        company_id: str,
        name: str,
        sector: str,
        is_fraud: bool = False,
        fraud_type: str = "None"
    ):
        self.company_id = company_id
        self.name = name
        self.sector = sector
        self.is_fraud = is_fraud
        self.fraud_type = fraud_type
        # Maps year (int) -> FinancialStatement
        self.statements: Dict[int, FinancialStatement] = {}

    def add_statement(self, statement: FinancialStatement) -> None:
        self.statements[statement.year] = statement

    def get_sorted_statements(self) -> List[FinancialStatement]:
        """Returns the annual statements sorted by year ascending."""
        return [self.statements[year] for year in sorted(self.statements.keys())]

    def to_flat_records(self) -> List[Dict[str, Any]]:
        """
        Converts the company metadata and annual financial statements into a flat list of dictionaries,
        suitable for loading into a Pandas DataFrame.
        """
        records = []
        for year, stmt in self.statements.items():
            record = {
                "company_id": self.company_id,
                "company_name": self.name,
                "sector": self.sector,
                "is_fraud": 1 if self.is_fraud else 0,
                "fraud_type": self.fraud_type,
            }
            # Merge financial metrics
            record.update(stmt.to_dict())
            records.append(record)
        return records
