import requests
import google.generativeai as genai
from config import GEMINI_API_KEY, SERPER_API_KEY
from typing import Tuple, Optional

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def search_news(query: str, num_results: int = 5) -> Optional[list]:
    """Use Serper API to search for recent news articles related to the claim."""
    url = "https://google.serper.dev/news"
    payload = {"q": query, "num": num_results}
    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        results = response.json()
        return results.get('news', [])
    except Exception as e:
        print(f"Error searching news: {e}")
        return None

def analyze_claim_with_evidence(claim: str, search_results: list) -> Tuple[str, str, str]:
    """Send the claim and the search results to Gemini for a verdict."""
    evidence_string = ""
    for idx, item in enumerate(search_results):
        evidence_string += f"Source {idx+1}: {item.get('title', 'No Title')}\n"
        evidence_string += f"Snippet: {item.get('snippet', 'No snippet available')}\n"
        evidence_string += f"Link: {item.get('link', 'No link')}\n\n"

    prompt = f"""
You are an expert fact-checking AI. Your task is to verify the following claim.

CLAIM: "{claim}"

Based ONLY on the provided search results from credible news sources, determine if the claim is TRUE, FALSE, or UNCERTAIN.

SEARCH RESULTS:
{evidence_string}

INSTRUCTIONS:
1. Provide your verdict strictly as "TRUE", "FALSE", or "UNCERTAIN".
2. Provide a confidence score (e.g., 95%) based on the strength of the evidence.
3. Provide a short, clear explanation justifying your verdict.

Output format:
VERDICT: (TRUE/FALSE/UNCERTAIN)
CONFIDENCE: (e.g., 95%)
EXPLANATION: (your explanation)
"""
    try:
        response = model.generate_content(prompt)
        response_text = response.text

        verdict = "UNCERTAIN"
        confidence = "0%"
        explanation = "Could not parse response."

        for line in response_text.split('\n'):
            if line.startswith("VERDICT:"):
                verdict = line.split("VERDICT:")[1].strip()
            elif line.startswith("CONFIDENCE:"):
                confidence = line.split("CONFIDENCE:")[1].strip()
            elif line.startswith("EXPLANATION:"):
                explanation = line.split("EXPLANATION:")[1].strip()

        return verdict, confidence, explanation
    except Exception as e:
        print(f"Error during Gemini analysis: {e}")
        return "ERROR", "0%", f"An error occurred: {str(e)}"