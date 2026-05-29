import os
import time
import traceback
from typing import List, Dict, Any

try:
    from groq import Groq
except ImportError:
    Groq = None
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"


# --- Structured Output Schema ---
class AuditQuestionsOutput(BaseModel):
    questions: List[str]


def _get_groq_client() -> Groq:
    if Groq is None:
        raise RuntimeError("Groq client is not installed. Set GROQ_API_KEY and install groq.")
    return Groq(api_key=GROQ_API_KEY)


def _groq_chat(prompt: str) -> str:
    """Helper: sends a single prompt to Groq and returns the text response."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured.")
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
        return "AI chat is temporarily unavailable."


def generate_audit_questions(anomalies: List[Dict[str, Any]], ratios: List[Dict[str, Any]]) -> List[str]:
    """
    Uses Groq (Llama 3.3 70B) to generate auditor-style questions based on detected anomalies.
    Returns a clean list of question strings.
    """
    if Groq is None or not GROQ_API_KEY:
        return generate_fallback_audit_questions(anomalies, ratios)

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


def generate_fallback_audit_questions(anomalies: List[Dict[str, Any]], ratios: List[Dict[str, Any]]) -> List[str]:
    questions = []

    if anomalies:
        first_anomaly = anomalies[0]
        description = first_anomaly.get("description", "the detected anomaly")
        questions.append(f"Can management explain the cause of {description}?")

    if ratios:
        first_ratio = ratios[0]
        ratio_name = first_ratio.get("name", "the reported ratio")
        questions.append(f"What supports the recent movement in {ratio_name}?")

    questions.append("What controls are in place to prevent similar issues in future periods?")

    return questions[:3]
