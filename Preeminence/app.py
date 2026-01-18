import streamlit as st
import pandas as pd
import re
import joblib
from pathlib import Path

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Genre Guesser 🎪",
    page_icon="🎧",
    layout="wide"
)

# ===================== CSS (Festival aesthetic) =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif !important;
}

/* Background */
.stApp {
    background: radial-gradient(circle at top,#1e0036,#12001e,#09000e 65%) !important;
    color: #eee !important;
}

/* Hero */
.hero {
    padding: 90px 20px;
    text-align: center;
    color: white;
}

.hero-title {
    font-size: 60px;
    font-weight: 900;
    letter-spacing: 2px;
    text-transform: uppercase;
    background: linear-gradient(90deg,#ff00ea,#ffcb3c,#00ffe7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
}

.hero-sub {
    font-size: 20px;
    margin-top: 10px;
    opacity: 0.85;
    color: white;
}

/* Input container */
.input-box {
    background: rgba(255,255,255,0.08);
    padding: 26px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 8px;
}

/* Text Area */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 2px solid rgba(255,255,255,0.2) !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 16px !important;
}

.stTextArea textarea:focus {
    border-color: #ff00ea !important;
    box-shadow: 0 0 15px rgba(255,0,234,0.3) !important;
}

.stTextArea label {
    color: white !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg,#ff3caa,#8b41ff) !important;
    color: white !important;
    font-size: 20px !important;
    border-radius: 40px !important;
    padding: 14px 32px !important;
    font-weight: 700 !important;
    border: none !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
}

.stButton>button:hover {
    transform: scale(1.03) !important;
    box-shadow: 0 0 22px rgba(255,60,186,0.5) !important;
}

/* Prediction Card */
.pred-card {
    margin-top: 24px;
    background: linear-gradient(135deg,#191035,#351051);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    animation: slideUp 0.5s ease-out;
}

@keyframes slideUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.pred-card h3 {
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 15px;
    color: #ffcb3c;
}

.pred-card h2 {
    font-size: 44px;
    font-weight: 900;
    background: linear-gradient(90deg,#ff00ea,#ffcb3c,#00ffe7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 20px 0;
}

.pred-card p {
    opacity: 0.8;
    font-size: 18px;
}

/* Example Chips */
.example-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    justify-content: center;
    margin: 20px 0;
}

.example-chip {
    background: rgba(255,255,255,0.1);
    padding: 10px 20px;
    border-radius: 25px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    border: 2px solid rgba(255,255,255,0.2);
    color: white;
}

.example-chip:hover {
    background: linear-gradient(135deg,#ff3caa,#8b41ff);
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(255,60,186,0.3);
}

/* Warning/Info */
.stWarning, .stInfo {
    background: rgba(255,203,60,0.15) !important;
    border: 1px solid rgba(255,203,60,0.3) !important;
    color: #ffcb3c !important;
    border-radius: 12px !important;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #ff00ea !important;
}

/* Footer */
.footer {
    margin-top: 40px;
    text-align: center;
    opacity: 0.45;
    font-size: 14px;
    padding-bottom: 10px;
    color: white;
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ===================== LOAD MODEL =====================
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

# ===================== TEXT CLEANING =====================
def clean_text(text):
    """Clean input text"""
    text = str(text).lower()
    return re.sub(r"[^a-z\s]", "", text)

# ===================== EXAMPLE LYRICS =====================
EXAMPLES = {
    "🎸 Rock": "Screaming guitars and thundering drums, headbanging all night long, metal rules supreme, power chords echoing through the night",
    "🎤 Hip-Hop": "Yo, dropping beats on the street, rhymes so sweet, can't be beat, rap game complete, flow so sick, lyrics quick",
    "🤠 Country": "Driving my truck down a dirt road, country music on the radio, simple life is all I know, boots and hat",
    "✨ Pop": "Baby you're the one for me, dancing all night feeling free, this love is meant to be, hearts collide under neon lights",
    "🎺 Blues": "I got the blues, feeling so down, my baby left me in this old town, walking these streets with nothing to lose",
}

# ===================== MAIN APP =====================
def main():
    # Load model
    model, tfidf, label_encoder = load_model()
    
    if model is None or tfidf is None or label_encoder is None:
        st.stop()
    
    # ===================== HERO SECTION =====================
    st.markdown("""
    <div class="hero">
        <div class="hero-title">GENRE GUESSER FESTIVAL</div>
        <div class="hero-sub">Submit lyrics • Get your stage • Enjoy the vibes 🎪</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Center column for better layout
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # ===================== EXAMPLE CHIPS =====================
        st.markdown('<div style="text-align: center; color: white; margin-bottom: 20px; font-size: 16px; font-weight: 600;">🎼 Try Example Lyrics:</div>', unsafe_allow_html=True)
        
        cols = st.columns(len(EXAMPLES))
        for idx, (label, example_text) in enumerate(EXAMPLES.items()):
            with cols[idx]:
                if st.button(label, key=f"example_{idx}", use_container_width=True):
                    st.session_state.selected_example = example_text
                    st.rerun()
        
        # ===================== INPUT BOX =====================
        st.markdown('<div class="input-box">', unsafe_allow_html=True)
        
        # Use selected example if available
        default_text = ""
        if 'selected_example' in st.session_state:
            default_text = st.session_state.selected_example
            st.session_state.pop('selected_example')
        
        lyrics_input = st.text_area(
            "🎤 Drop lyrics to join the lineup",
            value=default_text,
            placeholder="e.g. 'We were both young when I first saw you…'",
            height=200
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ===================== PREDICT ACTION =====================
        if st.button("Predict Genre 🎟️"):
            if len(lyrics_input.strip()) == 0:
                st.warning("bro send lyrics first 😭")
            else:
                with st.spinner("checking vibes… 🎧"):
                    cleaned = clean_text(lyrics_input)
                    X_test = tfidf.transform([cleaned])
                    pred = model.predict(X_test)[0]
                    pred_genre = label_encoder.inverse_transform([pred])[0]
                    
                    # Get confidence
                    probabilities = model.predict_proba(X_test)[0]
                    confidence = max(probabilities) * 100
                    
                    st.markdown(f"""
                    <div class="pred-card">
                        <h3>🎧 Genre Detected:</h3>
                        <h2>{pred_genre.upper()}</h2>
                        <p>This track belongs on the <b>{pred_genre}</b> stage 🤘</p>
                        <p style="margin-top: 15px; font-size: 16px;">Confidence: {confidence:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ===================== FOOTER =====================
    st.markdown("""
    <div class="footer">
    Made with ML • Music • Chaos
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

