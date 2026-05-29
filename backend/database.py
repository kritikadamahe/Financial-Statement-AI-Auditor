from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
# SQLALCHEMY_DATABASE_URL = "postgresql://user:password@postgresserver/db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def ensure_financial_records_schema() -> None:
    """Add any missing columns to the SQLite financial_records table."""
    inspector = inspect(engine)
    if not inspector.has_table("financial_records"):
        return

    existing_columns = {column["name"] for column in inspector.get_columns("financial_records")}
    required_columns = {
        "raw_parsed_data": "JSON",
        "industry_benchmarks": "JSON",
        "compliance_flags": "JSON",
    }

    with engine.begin() as connection:
        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                connection.execute(text(f"ALTER TABLE financial_records ADD COLUMN {column_name} {column_type}"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
