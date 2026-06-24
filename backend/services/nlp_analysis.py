import os
import google.generativeai as genai
from typing import Dict, Any, List

def analyze_mda_text(raw_text: str) -> Dict[str, Any]:
    """
    Analyzes the raw textual content (MD&A, footnotes) of a financial statement
    using Gemini to identify qualitative risk factors.
    """
    if not raw_text or len(raw_text.strip()) < 100:
        return {
            "risk_score": 0,
            "summary": "No significant text available for NLP analysis.",
            "flags": []
        }
        
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "risk_score": 0,
            "summary": "NLP analysis unavailable: GEMINI_API_KEY not configured.",
            "flags": []
        }
        
    genai.configure(api_key=api_key)
    
    # We use gemini-1.5-flash as it's fast and handles long contexts (up to 1M tokens)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    prompt = """
    You are an expert forensic accountant and auditor. Analyze the following extracted text from a financial statement (likely containing MD&A and footnotes).
    
    Look specifically for:
    1. Evasive or overly complex language regarding revenue recognition.
    2. Undisclosed or downplayed related-party transactions.
    3. Changes in accounting estimates or policies that inflate earnings.
    4. Going concern risks or liquidity warnings.
    
    Output a JSON object EXACTLY in this format, with no markdown code blocks:
    {
      "risk_score": <int between 0 and 100, where 100 is extremely suspicious>,
      "summary": "<A 2-3 sentence summary of the textual risk profile>",
      "flags": [
        "<flag 1>",
        "<flag 2>"
      ]
    }
    
    Financial Text:
    """
    
    # Truncate text if extremely long to save time, though Gemini 1.5 can handle a lot
    text_to_analyze = raw_text[:50000] 
    
    try:
        response = model.generate_content(prompt + text_to_analyze)
        response_text = response.text.strip()
        
        # Strip markdown if Gemini includes it
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        import json
        result = json.loads(response_text)
        return result
    except Exception as e:
        print(f"[NLP] Text analysis failed: {e}")
        return {
            "risk_score": 0,
            "summary": f"NLP analysis failed: {str(e)}",
            "flags": []
        }
