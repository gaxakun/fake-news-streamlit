import streamlit as st
import requests
import json
import google.generativeai as genai
from config import GEMINI_API_KEY, SERPER_API_KEY

# Configure Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def search_google_live(claim_text):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": claim_text, "num": 5})
    headers = {
        'X-API-KEY': SERPER_API_KEY,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(url, headers=headers, data=payload)
        return response.json().get("organic", [])
    except Exception:
        return []

# --- MOBILE RESPONSIVE UI CONFIGURATION ---
st.set_page_config(
    page_title="VerifyBot Mobile", 
    page_icon="🔍", 
    layout="centered"  # Centers the content perfectly on both laptops and phones
)

st.title("🔍 VerifyBot: Live Fact-Checker")
st.caption("Cross-references statements against real-time web documentation using Google Gemini.")

# text_area auto-adjusts size on mobile views
claim = st.text_area("What claim would you like to verify?", placeholder="e.g., There was a kidnapping incident in Nigeria today", height=100)

if st.button("Check Claim", type="primary", use_container_width=True):
    if not claim.strip():
        st.warning("Please type a claim first.")
    else:
        with st.spinner("Searching the internet for breaking news..."):
            sources = search_google_live(claim)
        # Then do something with 'sources' (like call Gemini analysis)
            sources = search_google_live(claim)
            
            if not sources:
                st.error("No real-time search results found for this specific wording. Try changing your search keywords.")
            else:
                context = ""
                for idx, src in enumerate(sources, 1):
                    context += f"[Source {idx}]\nTitle: {src.get('title')}\nSnippet: {src.get('snippet')}\n\n"
                
                system_prompt = (
                    "You are a strict fact-checker. Based ONLY on the provided web context, "
                    "return a JSON object with 'verdict' (TRUE/FALSE/UNCERTAIN), "
                    "'confidence' (0-100), and a short 'explanation' citing the sources."
                )
                user_prompt = f"Claim: {claim}\n\nWeb Results:\n{context}"
                
                try:
                    model = genai.GenerativeModel(
                        model_name="gemini-2.5-flash",
                        generation_config={"response_mime_type": "application/json"},
                        system_instruction=system_prompt
                    )
                    response = model.generate_content(user_prompt)
                    
                    # Clean up response string if markdown block wrappers exist
                    raw_text = response.text.strip()
                    if raw_text.startswith("```json"):
                        raw_text = raw_text.replace("```json", "", 1).replace("```", "", -1).strip()
                    elif raw_text.startswith("```"):
                        raw_text = raw_text.replace("```", "", 2).strip()
                        
                    result = json.loads(raw_text)
                    
                    st.divider()
                    verdict = result.get("verdict", "UNCERTAIN").upper()
                    confidence = result.get("confidence", "0")
                    
                    # Mobile friendly color-blocked cards
                    if "TRUE" in verdict:
                        st.success(f"**Verdict:** TRUE ({confidence}% Confidence) ✅")
                    elif "FALSE" in verdict:
                        st.error(f"**Verdict:** FALSE ({confidence}% Confidence) ❌")
                    else:
                        st.warning(f"**Verdict:** UNCERTAIN ({confidence}% Confidence) 🤔")
                        
                    st.markdown(f"**AI Explanation:** {result.get('explanation')}")
                    
                    # Collapsible layout section to save scrolling space on small mobile screens
                    with st.expander("🔗 View Traced Evidence Sources"):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(f"**[{idx}] [{src.get('title')}]({src.get('link')})**")
                            st.caption(f"\"{src.get('snippet')}\"")
                except Exception as e:
                    st.error(f"An interpretation error occurred: {e}")
