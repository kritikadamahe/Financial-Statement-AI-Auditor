from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List
from pydantic import BaseModel
import models
import schemas
from services.parser import parse_financial_statement
from services.analysis import extract_financial_metrics, calculate_ratios, detect_anomalies, get_industry_benchmarks, reconciliation_checks
from services.ai import generate_audit_questions, check_compliance, chat_with_financials
from services.nlp_analysis import analyze_mda_text
from database import engine, Base, get_db

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Auditr API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_origin_regex="https://.*\.vercel\.app.*|https://.*", # Allow Vercel and other https domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Financial Audit Prep API is running."}


@app.post("/api/upload", response_model=schemas.FinancialRecordResponse)
async def upload_file(
    files: List[UploadFile] = File(...), 
    industry: str = Form("Technology"),
    db: Session = Depends(get_db)
):
    try:
        all_metrics_by_year = {}
        filenames = []
        
        # 1. Parse Data (Multi-file support)
        for file in files:
            if not file.filename.endswith((".csv", ".xlsx", ".xls", ".pdf")):
                raise HTTPException(status_code=400, detail=f"Invalid file type for {file.filename}")
            filenames.append(file.filename)
            contents = await file.read()
            df, raw_text = parse_financial_statement(contents, file.filename)
            file_metrics = extract_financial_metrics(df)
            
            # Merge metrics
            for year, metrics in file_metrics.items():
                if year not in all_metrics_by_year:
                    all_metrics_by_year[year] = {}
                all_metrics_by_year[year].update(metrics)

        # 2. Analysis & Rules
        ratios = calculate_ratios(all_metrics_by_year)
        anomalies = detect_anomalies(all_metrics_by_year)
        reconciliation = reconciliation_checks(all_metrics_by_year)
        benchmarks = get_industry_benchmarks(industry)

        # 3. AI Layer
        nlp_analysis_result = analyze_mda_text(raw_text) if 'raw_text' in locals() else None

        # Apply NLP Veto if risk_score is low (< 30) AND NLP analysis actually succeeded (not failed or unavailable)
        if nlp_analysis_result and "NLP analysis unavailable" not in nlp_analysis_result.get('summary', '') and "NLP analysis failed" not in nlp_analysis_result.get('summary', ''):
            if nlp_analysis_result.get('risk_score', 100) < 30:
                filtered_anomalies = []
                for anomaly in anomalies:
                    severity = anomaly.get('severity', '')
                    if severity in ['Medium', 'Low']:
                        # Drop / Veto this anomaly
                        continue
                    elif severity == 'High':
                        # Downgrade to Medium
                        anomaly['severity'] = 'Medium'
                        filtered_anomalies.append(anomaly)
                    else:
                        filtered_anomalies.append(anomaly)
                anomalies = filtered_anomalies

        audit_questions = generate_audit_questions(anomalies, ratios)
        compliance_flags = check_compliance(all_metrics_by_year)

        raw_data_summary = {"years_analyzed": list(all_metrics_by_year.keys())}
        combined_filename = ", ".join(filenames)

        # 4. Persist to SQLite
        record = models.FinancialRecord(
            filename=combined_filename,
            status="Success",
            ratios=[r for r in ratios],
            anomalies=[a for a in anomalies],
            audit_questions=audit_questions,
            raw_data_summary=raw_data_summary,
            raw_parsed_data=all_metrics_by_year,
            industry_benchmarks=benchmarks,
            compliance_flags=compliance_flags,
            reconciliation_results=reconciliation,
            nlp_analysis=nlp_analysis_result
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return record

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history", response_model=list[schemas.HistoryListItem])
def get_history(db: Session = Depends(get_db)):
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
    record = db.query(models.FinancialRecord).filter(
        models.FinancialRecord.id == record_id
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Record {record_id} not found.")
    return record


class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat/{record_id}")
def chat_endpoint(record_id: int, req: ChatRequest, db: Session = Depends(get_db)):
    record = db.query(models.FinancialRecord).filter(
        models.FinancialRecord.id == record_id
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    
    response_text = chat_with_financials(
        query=req.query, 
        raw_data=record.raw_parsed_data or {}, 
        anomalies=record.anomalies or []
    )
    return {"reply": response_text}
