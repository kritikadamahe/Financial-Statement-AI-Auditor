import pandas as pd
import io

def parse_financial_statement(file_contents: bytes, filename: str) -> pd.DataFrame:
    """
    Parses an uploaded Excel or CSV file into a pandas DataFrame.
    Returns a cleaned DataFrame.
    """
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
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    
    # Basic cleaning
    df = df.dropna(how='all') # drop fully empty rows
    df = df.dropna(axis=1, how='all') # drop fully empty columns
    
    return df
