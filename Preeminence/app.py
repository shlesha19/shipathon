import streamlit as st
import re
import joblib
import os
import base64

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Genre Guesser",
    page_icon="🎧",
    layout="wide"
)

# ===================== PATHS =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    return model, tfidf, label_encoder

# ===================== TEXT CLEAN =====================
def clean_text(text):
    text = text.lower()
    return re.sub(r"[^a-z\s]", "", text)

# ===================== STARRY + GRADIENT UI =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ===== MAIN BACKGROUND ===== */
.stApp {
    background:
        radial-gradient(circle at 20% 20%, #1e90ff22, transparent 40%),
        radial-gradient(circle at 80% 30%, #4169e122, transparent 40%),
        radial-gradient(circle at 50% 80%, #0b3d9122, transparent 45%),
        linear-gradient(180deg, #050b1e, #081a3a, #050b1e);
    color: #eaf1ff;
    overflow-x: hidden;
}

/* ===== STAR FIELD ===== */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(1px 1px at 20% 30%, white, transparent),
        radial-gradient(1px 1px at 80% 20%, white, transparent),
        radial-gradient(1px 1px at 50% 70%, white, transparent),
        radial-gradient(1px 1px at 10% 90%, white, transparent),
        radial-gradient(1px 1px at 90% 80%, white, transparent);
    background-repeat: repeat;
    background-size: 300px 300px;
    opacity: 0.25;
    z-index: 0;
    animation: starsMove 120s linear infinite;
}

@keyframes starsMove {
    from { transform: translateY(0); }
    to { transform: translateY(-2000px); }
}

/* ===== HERO ===== */
.hero {
    padding: 110px 20px 60px;
    text-align: center;
}

.hero-title {
    font-size: 58px;
    font-weight: 800;
    background: linear-gradient(90deg, #8ec5ff, #5a7dff, #8ec5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    margin-top: 16px;
    font-size: 19px;
    max-width: 720px;
    margin-inline: auto;
    color: #c7d8ff;
    line-height: 1.6;
}

/* ===== GLASS CARD ===== */
.glass {
    background: rgba(10, 20, 50, 0.6);
    border-radius: 22px;
    padding: 32px;
    border: 1px solid rgba(140,170,255,0.18);
    backdrop-filter: blur(18px);
    box-shadow: 0 30px 80px rgba(0,0,0,0.45);
}

/* ===== TEXT AREA ===== */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(140,170,255,0.35) !important;
    color: #eef3ff !important;
    font-size: 16px !important;
}

.stTextArea textarea:focus {
    border-color: #7aa2ff !important;
    box-shadow: 0 0 0 3px rgba(122,162,255,0.25) !important;
}

/* ===== BUTTON ===== */
.stButton > button {
    background: linear-gradient(135deg, #6aa6ff, #4b6cff);
    color: white !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    border-radius: 999px !important;
    padding: 14px !important;
    width: 100%;
    border: none !important;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 16px 40px rgba(90,125,255,0.45);
}

/* ===== RESULT CARD ===== */
.result {
    margin-top: 34px;
    text-align: center;
    padding: 36px;
    border-radius: 26px;
    background: linear-gradient(
        180deg,
        rgba(40,80,180,0.35),
        rgba(15,30,80,0.65)
    );
    border: 1px solid rgba(140,170,255,0.25);
    backdrop-filter: blur(20px);
}

.result h2 {
    font-size: 46px;
    font-weight: 800;
    background: linear-gradient(90deg,#a3c9ff,#ffffff,#a3c9ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ===== FOOTER ===== */
.footer {
    text-align: center;
    opacity: 0.45;
    font-size: 14px;
    padding: 60px 0 20px;
}

/* Hide Streamlit UI */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

</style>
""", unsafe_allow_html=True)

# ===================== EXAMPLES =====================
EXAMPLES = {
    "Rock": "Crashing drums, distorted guitars, screaming into the night",
    "Hip-Hop": "Rhymes and rhythm, stories from the street, beats hitting hard",
    "Pop": "Bright lights, dancing hearts, everything feels alive tonight",
    "Country": "Dusty roads, old trucks, sunsets and memories",
    "Blues": "Slow nights, heavy hearts, melodies soaked in pain"
}

# ===================== APP =====================
def main():
    model, tfidf, label_encoder = load_model()

    st.markdown("""
    <div class="hero">
        <div class="hero-title">Genre Guesser</div>
        <div class="hero-sub">
            Music carries patterns hidden in words.  
            Paste a few lines of lyrics and let the model
            quietly infer the genre.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.markdown("<div class='glass'>", unsafe_allow_html=True)

        cols = st.columns(len(EXAMPLES))
        for i, (g, txt) in enumerate(EXAMPLES.items()):
            with cols[i]:
                if st.button(g):
                    st.session_state.sample = txt
                    st.rerun()

        lyrics = st.text_area(
            "Lyrics",
            value=st.session_state.pop("sample", ""),
            placeholder="Type or paste a few lines here…",
            height=200
        )

        if st.button("Predict genre"):
            if lyrics.strip() == "":
                st.warning("Please enter some lyrics.")
            else:
                cleaned = clean_text(lyrics)
                X = tfidf.transform([cleaned])
                pred = model.predict(X)[0]
                genre = label_encoder.inverse_transform([pred])[0]
                confidence = max(model.predict_proba(X)[0]) * 100

                st.markdown(f"""
                <div class="result">
                    <h3>Predicted genre</h3>
                    <h2>{genre.upper()}</h2>
                    <p>Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='footer'>Built with ML • music • curiosity</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
