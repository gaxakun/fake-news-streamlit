import streamlit as st
import pandas as pd
import numpy as np
import joblib
from utils.preprocess import preprocess_text

st.set_page_config(page_title="🛡️ FakeShield AI", page_icon="🛡️", layout="wide")

# ========== SIDEBAR DARK MODE ==========
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    dark_mode = st.toggle("🌙 Dark Mode", value=False)
    st.markdown("---")
    st.markdown("### 🧠 Models")
    st.markdown("- Logistic Regression")
    st.markdown("- Naive Bayes")
    st.markdown("- Passive Aggressive")
    st.markdown("- Ensemble Voting")
    st.markdown("---")
    st.caption("Trained on 44k+ ISOT dataset (5000 features)")

# ========== DARK MODE CSS ==========
if dark_mode:
    css = """
    <style>
        .stApp { background-color: #0e1117; }
        body, .stMarkdown, .stSubheader, h1, h2, h3, h4, h5, h6, p, li, span, div:not(.stAlert) {
            color: #f0f2f6 !important;
        }
        .main-header { color: #9bb7d4 !important; text-align: center; }
        .sub-header { color: #cccccc !important; text-align: center; }
        .result-box.fake { background: #2a1a1a; border-left-color: #ff6666; }
        .result-box.real { background: #1a2a1a; border-left-color: #66ff66; }
        .model-card { background: #1e1e2e; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
        .model-card h4, .model-card p, .model-card small { color: #f0f2f6 !important; }
        .stTextArea textarea, .stTextInput input {
            background-color: #1e1e2e !important;
            color: #f0f2f6 !important;
            border: 1px solid #444 !important;
        }
        .stTextArea textarea::placeholder, .stTextInput input::placeholder {
            color: #888 !important;
        }
        .stButton button { background-color: #4F7EF7; color: white; }
        .stButton button[kind="secondary"] { background-color: #333; border-color: #555; color: #ddd; }
        hr { border-color: #333; }
        [data-testid="stSidebar"] { background-color: #0a0c10; }
        [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
        .stCaption, footer, .caption { color: #aaa !important; }
        .stAlert { background-color: #1e2a2a !important; color: #ddd !important; }
        .stDivider, hr { display: none !important; }
    </style>
    """
else:
    css = """
    <style>
        .main-header { color: #1E3A5F; text-align: center; }
        .sub-header { text-align: center; }
        .result-box.fake { background: #ffe0e0; }
        .result-box.real { background: #e0ffe0; }
        .model-card { background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.1); }
        .stDivider, hr { display: none !important; }
    </style>
    """
st.markdown(css, unsafe_allow_html=True)

# ========== RESPONSIVE CONTAINER ==========
st.markdown("""
<style>
    .main .block-container { max-width: 1100px; padding: 2rem 1.5rem; margin: 0 auto; }
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0.25rem; }
    .sub-header { margin-bottom: 1.5rem; font-size: 0.95rem; }
    .result-box { padding: 1.2rem; border-radius: 12px; margin: 1rem 0; text-align: center; }
    .model-card { flex: 1; min-width: 180px; padding: 1rem; border-radius: 10px; text-align: center; background: inherit; }
    @media (max-width: 700px) { .model-card { min-width: 100%; } .main-header { font-size: 1.6rem; } }
    .stButton button { border-radius: 8px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ========== LOAD MODELS ==========
@st.cache_resource
def load_models():
    try:
        vec = joblib.load('models/vectorizer.pkl')
        lr = joblib.load('models/lr_model.pkl')
        nb = joblib.load('models/nb_model.pkl')
        pac = joblib.load('models/pac_model.pkl')
        return vec, lr, nb, pac, True
    except Exception as e:
        return None, None, None, None, False

vectorizer, lr_model, nb_model, pac_model, ok = load_models()
if not ok:
    st.error("❌ Models not found. Run `python train_model.py` first.")
    st.stop()

# ========== PREDICTION FUNCTION ==========
def predict(text):
    cleaned = preprocess_text(text)
    X = vectorizer.transform([cleaned])
    lr_pred = lr_model.predict(X)[0]
    lr_prob = lr_model.predict_proba(X)[0]
    nb_pred = nb_model.predict(X)[0]
    nb_prob = nb_model.predict_proba(X)[0]
    pac_pred = pac_model.predict(X)[0]

    lr_fake_idx = 0 if lr_model.classes_[0] == 'FAKE' else 1
    nb_fake_idx = 0 if nb_model.classes_[0] == 'FAKE' else 1

    results = {
        'Logistic Regression': {
            'pred': lr_pred,
            'conf': max(lr_prob) * 100,
            'fake': lr_prob[lr_fake_idx] * 100,
            'real': lr_prob[1-lr_fake_idx] * 100
        },
        'Naive Bayes': {
            'pred': nb_pred,
            'conf': max(nb_prob) * 100,
            'fake': nb_prob[nb_fake_idx] * 100,
            'real': nb_prob[1-nb_fake_idx] * 100
        },
        'Passive Aggressive': {
            'pred': pac_pred,
            'conf': 100,
            'fake': 100 if pac_pred == 'FAKE' else 0,
            'real': 100 if pac_pred == 'REAL' else 0
        }
    }
    fake_votes = sum(1 for r in results.values() if r['pred'] == 'FAKE')
    real_votes = sum(1 for r in results.values() if r['pred'] == 'REAL')
    results['Ensemble'] = {
        'pred': 'FAKE' if fake_votes > real_votes else 'REAL',
        'conf': max(fake_votes, real_votes) / 3 * 100,
        'fake': fake_votes / 3 * 100,
        'real': real_votes / 3 * 100
    }
    return results

# ========== UI ==========
st.markdown("<div class='main-header'>🛡️ FakeShield AI</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Comparative Analysis · Logistic Regression · Naive Bayes · Passive Aggressive · Ensemble</div>", unsafe_allow_html=True)

st.subheader("📝 Enter any news text")

# Session state for input
if 'input_text' not in st.session_state:
    st.session_state.input_text = ""

user_input = st.text_area(
    "Paste a news headline or article:",
    value=st.session_state.input_text,
    height=120,
    placeholder="e.g., 'President Tinubu signs new budget...'",
    key="text_area_widget"
)

if user_input != st.session_state.input_text:
    st.session_state.input_text = user_input

col1, col2 = st.columns([4, 1])
with col1:
    detect_clicked = st.button("🔍 Detect Fake News", type="primary", use_container_width=True)
with col2:
    clear_clicked = st.button("🗑️ Clear", use_container_width=True)

if clear_clicked:
    st.session_state.input_text = ""
    st.session_state.pop("results", None)
    st.rerun()

if detect_clicked and st.session_state.input_text:
    text_to_analyze = st.session_state.input_text.strip()
    if len(text_to_analyze) < 20:
        st.info("💡 Short text may reduce accuracy. For best results, paste a full article.")
    if text_to_analyze:
        with st.spinner("🧠 Analyzing with all 3 models..."):
            results = predict(text_to_analyze)
        st.session_state.results = results
        st.session_state.text_len = len(text_to_analyze)
        st.session_state.last_input = st.session_state.input_text
        st.rerun()

# ========== DISPLAY RESULTS ==========
if 'results' in st.session_state:
    res = st.session_state.results
    ensemble = res['Ensemble']
    verdict_icon = "🚨" if ensemble['pred'] == 'FAKE' else "✅"

    if ensemble['conf'] < 60:
        st.warning(f"""
        ### ⚠️ LOW CONFIDENCE ({ensemble['conf']:.0f}%)
        The models are uncertain. This text might be **real news written in a sensational style**.
        **Verdict not definitive.** Please manually check the source.
        """)
    else:
        verdict_class = "fake" if ensemble['pred'] == 'FAKE' else "real"
        st.markdown(f"""
        <div class='result-box {verdict_class}'>
            <h3>{verdict_icon} FINAL VERDICT: {ensemble['pred']} NEWS</h3>
            <p>Ensemble confidence: <strong>{ensemble['conf']:.1f}%</strong> | 
            FAKE: {ensemble['fake']:.0f}% · REAL: {ensemble['real']:.0f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # Override checkbox (no rerun to avoid blur)
    override = st.checkbox("📢 Override to REAL (demo only)")
    if override:
        ensemble['pred'] = 'REAL'
        ensemble['conf'] = 95
        st.success("✅ Override active – verdict forced to REAL.")

    st.markdown("### 📊 Model Predictions")
    cols = st.columns(3)
    models_list = ['Logistic Regression', 'Naive Bayes', 'Passive Aggressive']
    for i, name in enumerate(models_list):
        r = res[name]
        if dark_mode:
            pred_color = "#ff6666" if r['pred'] == 'FAKE' else "#66ff66"
        else:
            pred_color = "#ff4444" if r['pred'] == 'FAKE' else "#44aa44"
        with cols[i]:
            st.markdown(f"""
            <div class='model-card'>
                <h4>{name}</h4>
                <h2 style='color: {pred_color};'>{r['pred']}</h2>
                <progress value="{r['conf']}" max="100" style="width:100%; height:8px;"></progress>
                <p>Confidence: {r['conf']:.1f}%</p>
                <small>Fake {r['fake']:.0f}% · Real {r['real']:.0f}%</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("### 📈 Text Statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Characters", st.session_state.text_len)
    last_input = st.session_state.get('last_input', '')
    word_count = len(last_input.split()) if last_input else 0
    c2.metric("Words", word_count)
    c3.metric("Models Used", 3)

    if st.button("Clear Results", type="secondary"):
        st.session_state.pop("results", None)
        st.rerun()

st.caption("🛡️ FakeShield AI | Trained on 44k+ news articles (ISOT dataset) | Text analysis only")