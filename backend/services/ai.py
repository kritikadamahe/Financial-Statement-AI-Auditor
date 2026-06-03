import os
import time
import traceback
from typing import List, Dict, Any

from groq import Groq
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load .env from the backend/ directory (parent of services/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

GROQ_MODEL = "llama-3.3-70b-versatile"


# --- Structured Output Schema ---
class AuditQuestionsOutput(BaseModel):
    questions: List[str]


def _get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return Groq(api_key=api_key)


def _groq_chat(prompt: str) -> str:
    """Helper: sends a single prompt to Groq and returns the text response."""
    client = _get_groq_client()
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GROQ_MODEL,
        temperature=0.4,
        max_tokens=1024,
    )
    return chat_completion.choices[0].message.content


def check_compliance(raw_data_summary: Dict[str, Any]) -> List[str]:
    """
    Validates financial format against GAAP/IFRS using Groq.
    """
    prompt = (
        "You are an expert CPA. Review the following financial statement metrics "
        "and list any potential GAAP/IFRS compliance flags or missing standard line items.\n"
        f"{raw_data_summary}\n\n"
        "Return 2-3 short bullet points only. Do not use asterisks or formatting."
    )

    try:
        text = _groq_chat(prompt)
        flags = [f.strip().lstrip("-").strip() for f in text.split("\n") if f.strip()]
        return flags
    except Exception:
        return ["Unable to perform compliance check at this time."]


def chat_with_financials(query: str, raw_data: Dict[str, Any], anomalies: List[Dict[str, Any]]) -> str:
    """
    Answers a natural language query based on the uploaded spreadsheet data using Groq.
    """
    prompt = (
        "You are a financial analyst copilot. Answer the user's question based strictly on the provided financial data.\n"
        f"Data: {raw_data}\n"
        f"Anomalies: {anomalies}\n\n"
        f"User Question: {query}"
    )

    try:
        return _groq_chat(prompt)
    except Exception as e:
        return f"Error analyzing data: {e}"


def generate_audit_questions(anomalies: List[Dict[str, Any]], ratios: List[Dict[str, Any]]) -> List[str]:
    """
    Uses Groq (Llama 3.3 70B) to generate auditor-style questions based on detected anomalies.
    Returns a clean list of question strings.
    """
    # Build a rich prompt with full context
    prompt = (
        "You are a senior financial auditor reviewing a company's financial statements. "
        "Below are anomalies and financial ratios that were automatically detected.\n\n"
    )

    if anomalies:
        prompt += "=== Detected Anomalies ===\n"
        for idx, anomaly in enumerate(anomalies):
            prompt += f"{idx + 1}. [{anomaly['severity']} Risk] {anomaly['description']}\n"
    else:
        prompt += "=== Detected Anomalies ===\nNo major anomalies were detected.\n"

    if ratios:
        prompt += "\n=== Computed Financial Ratios ===\n"
        for ratio in ratios:
            prompt += f"- {ratio['name']} ({ratio['year']}): {ratio['value']}\n"

    prompt += (
        "\nBased on these findings, generate 3 to 5 critical questions you would ask "
        "management during an audit. Make questions specific to the anomalies and ratios "
        "provided above. Return only the questions — no preamble, no numbering."
    )

    for attempt in range(3):
        try:
            text = _groq_chat(prompt)

            questions = [
                q.strip().lstrip("-").strip()
                for q in text.split("\n")
                if q.strip()
            ]

            return questions

        except Exception as e:
            if attempt < 2:
                print(f"Groq call failed (Attempt {attempt + 1}/3), retrying in 2s: {e}")
                time.sleep(2)
            else:
                print(f"Groq call failed after 3 attempts: {e}")
                traceback.print_exc()
                return ["Failed to generate questions due to AI service unavailability. Please try again later."]


def extract_csv_from_pdf_text(raw_text: str) -> str:
    """
    Uses Groq to extract structured financial data from raw PDF text.
    Returns a clean CSV formatted string.
    """
    prompt = (
        "You are an expert financial data extractor. I am giving you raw text extracted from a PDF financial report. "
        "Extract ALL financial metrics you can find and output them as a CSV.\n\n"
        "STRICT RULES:\n"
        "1. Output ONLY valid CSV text. No markdown, no explanation, no ```csv blocks.\n"
        "2. First column header MUST be 'Metric'. Remaining column headers MUST be the actual fiscal years as 4-digit numbers (e.g., 2022,2023,2024).\n"
        "3. Use EXACTLY these metric names where applicable:\n"
        "   - Revenue\n"
        "   - Cost of Goods Sold\n"
        "   - Gross Profit\n"
        "   - Total Expenses\n"
        "   - Net Income\n"
        "   - Current Assets\n"
        "   - Total Assets\n"
        "   - Current Liabilities\n"
        "   - Total Debt\n"
        "   - Total Equity\n"
        "   - Operating Cash Flow\n"
        "   - Inventory\n"
        "   - Accounts Receivable\n"
        "4. All numbers must be plain integers or floats. No commas, no dollar signs, no parentheses.\n"
        "5. Use negative signs for negative numbers.\n\n"
        "EXAMPLE OUTPUT:\n"
        "Metric,2022,2023,2024\n"
        "Revenue,45000,62000,89000\n"
        "Cost of Goods Sold,13500,17360,24920\n"
        "Net Income,9000,13640,21380\n\n"
        f"RAW TEXT:\n\n{raw_text[:8000]}"
    )

    try:
        client = _get_groq_client()
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=GROQ_MODEL,
            temperature=0.2,
            max_tokens=2048,
        )
        text = chat_completion.choices[0].message.content
        # Clean up in case the model added markdown despite instructions
        if "```" in text:
            # Extract content between ``` blocks
            lines = text.split("\n")
            clean_lines = []
            inside_block = False
            for line in lines:
                if line.strip().startswith("```"):
                    inside_block = not inside_block
                    continue
                if inside_block or not text.startswith("```"):
                    clean_lines.append(line)
            text = "\n".join(clean_lines)
        
        text = text.strip()
        print(f"[PDF Extract] Groq returned CSV:\n{text[:500]}")
        return text
    except Exception as e:
        print(f"Failed to extract CSV from PDF: {e}")
        raise ValueError("Failed to extract structured data from PDF via AI.")
