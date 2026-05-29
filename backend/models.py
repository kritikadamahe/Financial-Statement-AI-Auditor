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
