from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class FinancialRecordBase(BaseModel):
    filename: str


class FinancialRecordCreate(FinancialRecordBase):
    pass


class FinancialRecord(FinancialRecordBase):
    id: int
    upload_date: datetime
    status: str

    class Config:
        from_attributes = True


class RatioResult(BaseModel):
    name: str
    value: float
    year: str


class AnomalyResult(BaseModel):
    description: str
    severity: str  # e.g., "High", "Medium", "Low"
    related_metrics: List[str]


class AuditAnalysisResult(BaseModel):
    ratios: List[RatioResult]
    anomalies: List[AnomalyResult]
    audit_questions: List[str]
    raw_data_summary: Dict[str, Any]


# --- History Schemas ---

class HistoryListItem(BaseModel):
    """Lightweight schema for sidebar list — only metadata, no heavy payload."""
    id: int
    filename: str
    upload_date: datetime
    status: str
    anomaly_count: int
    ratio_count: int

    class Config:
        from_attributes = True


class FinancialRecordResponse(BaseModel):
    """Full schema for a single historical audit record."""
    id: int
    filename: str
    upload_date: datetime
    status: str
    ratios: Optional[List[RatioResult]] = []
    anomalies: Optional[List[AnomalyResult]] = []
    audit_questions: Optional[List[str]] = []
    raw_data_summary: Optional[Dict[str, Any]] = {}
    
    # New V2 Features
    raw_parsed_data: Optional[Dict[str, Any]] = None
    industry_benchmarks: Optional[Dict[str, Any]] = None
    compliance_flags: Optional[List[str]] = []

    class Config:
        from_attributes = True
