import streamlit as st
import joblib
import re

import os
import streamlit as st
import joblib

@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

    return model, tfidf, label_encoder

# ----------------------------------------------------
# TEXT CLEANER
# ----------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    return re.sub(r"[^a-z\s]", "", text)

# Load models once
model, tfidf, label_encoder = load_models()

# ----------------------------------------------------
# UI CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="Genre Guesser 🎶✨",
    page_icon="🎧",
    layout="centered"
)

# Gen-Z CSS theme
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#ffedff,#d6e9ff,#ffe8f3);
}
.big-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    padding: 12px;
}
.subtle {
    font-size: 16px;
    text-align:center;
    color: #555;
    margin-bottom: 12px;
}
.predict-box {
    background: rgba(255,255,255,0.55);
    padding: 18px;
    border-radius: 14px;
    backdrop-filter: blur(10px);
    margin-top: 14px;
    border: 1px solid rgba(255,255,255,0.45);
}
.stButton>button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    font-size: 18px;
    font-weight: 600;
    padding: 12px 30px;
    border-radius: 50px;
    border: none;
    width: 100%;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# TITLE
# ----------------------------------------------------
st.markdown('<div class="big-title">🎶 Genre Guesser 3000 ✨</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Paste song lyrics. I\'ll vibe-check the genre 😎</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# USER INPUT
# ----------------------------------------------------
lyrics_input = st.text_area(
    "🎤 Drop lyrics here",
    placeholder="e.g. 'We were both young when I first saw you…'",
    height=180
)

# ----------------------------------------------------
# PREDICT ACTION
# ----------------------------------------------------
if st.button("Predict Genre 🔮"):
    if len(lyrics_input.strip()) == 0:
        st.warning("bruh type something first 😭")
    else:
        with st.spinner("vibe-checking… 🧠✨"):
            cleaned = clean_text(lyrics_input)
            X_test = tfidf.transform([cleaned])
            pred = model.predict(X_test)[0]
            pred_genre = label_encoder.inverse_transform([pred])[0]
            
            st.markdown(f"""
            <div class='predict-box'>
                <h3>🎧 Vibe detected:</h3>
                <h2><b>{pred_genre}</b></h2>
                <p>Certified genre moment 💅</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show top 3 predictions if
