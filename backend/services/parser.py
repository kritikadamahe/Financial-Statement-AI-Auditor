import pandas as pd
import io
import fitz  # PyMuPDF
from services.ai import extract_csv_from_pdf_text

from typing import Tuple

def parse_financial_statement(file_contents: bytes, filename: str) -> Tuple[pd.DataFrame, str]:
    """
    Parses an uploaded Excel, CSV, or PDF file into a pandas DataFrame.
    Also returns the raw extracted text (useful for NLP analysis of PDFs).
    """
    raw_text = ""
    if filename.endswith(".csv"):
        # Try different encodings, as files can come from Excel (UTF-8 BOM), PowerShell (UTF-16), etc.
        encodings = ['utf-8-sig', 'utf-16', 'utf-8', 'cp1252']
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(io.BytesIO(file_contents), encoding=enc)
                # If we got more than 1 column, it likely parsed correctly
                if len(df.columns) > 1:
                    break
            except Exception:
                continue
                
        if df is None or len(df.columns) <= 1:
            # Fallback to default if all failed or only found 1 column
            df = pd.read_csv(io.BytesIO(file_contents), encoding='utf-8', sep=None, engine='python')
            
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(file_contents))
    elif filename.endswith(".pdf"):
        # Extract text from PDF
        try:
            doc = fitz.open(stream=file_contents, filetype="pdf")
            for page in doc:
                raw_text += page.get_text() + "\n\n"
            
            # Use Groq to extract structured CSV
            csv_text = extract_csv_from_pdf_text(raw_text)
            
            # Load into Pandas robustly
            lines = [line for line in csv_text.split('\n') if line.strip()]
            if lines:
                header = lines[0].split(',')
                n_cols = len(header)
                clean_lines = [lines[0]]
                for line in lines[1:]:
                    parts = line.split(',')
                    if len(parts) > n_cols:
                        clean_lines.append(','.join(parts[:n_cols]))
                    elif len(parts) > 0:
                        parts.extend([''] * (n_cols - len(parts)))
                        clean_lines.append(','.join(parts))
                df = pd.read_csv(io.StringIO('\n'.join(clean_lines)))
            else:
                df = pd.DataFrame()
            
        except Exception as e:
            raise ValueError(f"Failed to process PDF: {e}")
    else:
        raise ValueError("Unsupported file format. Please upload a CSV, Excel, or PDF file.")
    
    # Basic cleaning
    df = df.dropna(how='all') # drop fully empty rows
    df = df.dropna(axis=1, how='all') # drop fully empty columns
    
    return df, raw_text
