from sqlalchemy import Column, Integer, String, DateTime, JSON
from database import Base
import datetime


class FinancialRecord(Base):
    __tablename__ = "financial_records"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default="Processing")

    # Analysis results stored as JSON for lightweight MVP persistence
    ratios = Column(JSON, nullable=True)           # List of ratio dicts
    anomalies = Column(JSON, nullable=True)        # List of anomaly dicts
    audit_questions = Column(JSON, nullable=True)  # List of question strings
    raw_data_summary = Column(JSON, nullable=True) # Dict with metadata (e.g. years_analyzed)
    
    # New V2 Features
    raw_parsed_data = Column(JSON, nullable=True)  # Store raw DF dict for chat
    industry_benchmarks = Column(JSON, nullable=True) # Benchmark dict
    compliance_flags = Column(JSON, nullable=True) # List of compliance violation strings
    
    # V3: Reconciliation Engine
    reconciliation_results = Column(JSON, nullable=True) # Dict of identical mapping checks
    
    # V4: NLP MD&A Risk Scoring
    nlp_analysis = Column(JSON, nullable=True) # Dict with risk_score, summary, flags dicts
