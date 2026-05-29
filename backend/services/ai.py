import os
import time
import traceback
from typing import List, Dict, Any

import google.generativeai as genai
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


# --- Structured Output Schema ---
# Defining this schema forces Gemini to return a valid JSON object with a
# "questions" key that holds a list of strings.  No newline-splitting needed.
class AuditQuestionsOutput(BaseModel):
    questions: List[str]


def generate_audit_questions(anomalies: List[Dict[str, Any]], ratios: List[Dict[str, Any]]) -> List[str]:
    """
    Uses Gemini to generate auditor-style questions based on detected anomalies.
    Returns a clean list of question strings via structured JSON output.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found. Skipping AI generation.")
        return ["AI integration is disabled. Please set GEMINI_API_KEY."]

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
            genai.configure(api_key=api_key)

            model = genai.GenerativeModel('gemini-1.5-flash')

            response = model.generate_content(prompt)

            questions = [
                q.strip().lstrip("-").strip()
                for q in response.text.split("\n")
                if q.strip()
            ]

            return questions

        except Exception as e:
            if attempt < 2:
                print(f"Gemini call failed (Attempt {attempt + 1}/3), retrying in 2s: {e}")
                time.sleep(2)
            else:
                print(f"Gemini call failed after 3 attempts: {e}")
                traceback.print_exc()
                return ["Failed to generate questions due to AI service unavailability. Please try again later."]
