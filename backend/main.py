from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import models
import schemas
from services.parser import parse_financial_statement
from services.analysis import extract_financial_metrics, calculate_ratios, detect_anomalies
from services.ai import generate_audit_questions
from database import engine, Base, get_db, ensure_financial_records_schema

# Create tables (safe to run repeatedly — only creates if not existing)
Base.metadata.create_all(bind=engine)
ensure_financial_records_schema()

app = FastAPI(title="Financial Audit Prep API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Financial Audit Prep API is running."}


@app.post("/api/upload", response_model=schemas.FinancialRecordResponse)
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV or Excel file.")

    try:
        contents = await file.read()

        # 1. Parse Data
        df = parse_financial_statement(contents, file.filename)

        # 2. Analysis & Rules
        metrics_by_year = extract_financial_metrics(df)
        ratios = calculate_ratios(metrics_by_year)
        anomalies = detect_anomalies(metrics_by_year)

        # 3. AI Layer
        audit_questions = generate_audit_questions(anomalies, ratios)

        raw_data_summary = {"years_analyzed": list(metrics_by_year.keys())}

        # 4. Persist to SQLite
        record = models.FinancialRecord(
            filename=file.filename,
            status="Success",
            ratios=[r for r in ratios],
            anomalies=[a for a in anomalies],
            audit_questions=audit_questions,
            raw_data_summary=raw_data_summary,
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return record

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history", response_model=list[schemas.HistoryListItem])
def get_history(db: Session = Depends(get_db)):
    """
    Returns all past audit records ordered by most recent first.
    Each item is lightweight — suitable for rendering a sidebar list.
    """
    records = db.query(models.FinancialRecord).order_by(
        models.FinancialRecord.upload_date.desc()
    ).all()

    result = []
    for r in records:
        result.append(schemas.HistoryListItem(
            id=r.id,
            filename=r.filename,
            upload_date=r.upload_date,
            status=r.status,
            anomaly_count=len(r.anomalies) if r.anomalies else 0,
            ratio_count=len(r.ratios) if r.ratios else 0,
        ))
    return result


@app.get("/api/history/{record_id}", response_model=schemas.FinancialRecordResponse)
def get_history_record(record_id: int, db: Session = Depends(get_db)):
    """
    Returns the full analysis payload for a specific historic record.
    Used by the frontend when the user clicks a sidebar entry.
    """
    record = db.query(models.FinancialRecord).filter(
        models.FinancialRecord.id == record_id
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Record {record_id} not found.")
    return record
