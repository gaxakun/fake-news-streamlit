# VerifyBot: Real-Time AI Fact-Checking System

VerifyBot is a dynamic, mobile-responsive web application designed to counter fake news and misinformation by cross-referencing claims against live web data using Generative AI.

Unlike traditional static machine learning models, VerifyBot connects directly to search engines to fetch current breaking news, assessing factual validity with real-time tracking metrics.

---

## Core Features

- **Live Context Retrieval:** Utilizes the Serper API to query Google News data instantly.
- **AI-Powered Evaluation:** Leverages Google Gemini models (`gemini-2.5-flash`) to comprehensively read, compare, and parse truth verification.
- **Transparent Evidence Sourcing:** Provides clickable anchor links and snippet traces for every verification step.
- **Mobile Responsive Design:** Engineered with a centered Streamlit layout optimization tailored for cross-platform utility.

---

## Project Structure

```text
├── .env                  # Secure environment credentials (Hidden locally)
├── config.py             # Global secret configuration and loader
├── fact_checker.py       # Core retrieval pipeline and Gemini analytical prompt
├── verify_bot.py         # Mobile-responsive Streamlit interface (Main Entrance)
└── requirements.txt      # Production library dependencies