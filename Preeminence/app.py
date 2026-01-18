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

# ===================== HELPERS =====================
def get_base64_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
frame1_b64 = get_base64_image(os.path.join(BASE_DIR, "frame1.png"))
frame2_b64 = get_base64_image(os.path.join(BASE_DIR, "frame2.png"))

# ===================== STYLES =====================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif !important;
}}

.stApp {{
    background: linear-gradient(180deg,#0b1d3a,#0a2a5c,#06152e);
    color: #eaf2ff;
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background-image: url("data:image/png;base64,{frame1_b64}");
    background-size: cover;
    background-position: center;
    opacity: 0.25;
    z-index: 0;
}}

.stApp::after {{
    content: "";
    position: fixed;
    inset: 0;
    background-image: url("data:image/png;base64,{frame2_b64}");
    background-size: cover;
    background-position: center;
    opacity: var(--scroll-opacity, 0);
    transition: opacity 0.4s ease;
    z-index: 0;
}}

.hero {{
    padding: 90px 20px 40px;
    text-align: center;
}}

.hero-title {{
    font-size: 54px;
    font-weight: 800;
    letter-spacing: -1px;
}}

.hero-sub {{
    margin-top: 14px;
    font-size: 18px;
    color: #b7ccff;
    max-width: 720px;
    margin-inline: auto;
}}

.input-box {{
    background: rgba(10,25,60,0.8);
    padding: 28px;
    border-radius: 20px;
    border: 1px solid rgba(160,190,255,0.18);
    backdrop-filter: blur(14px);
}}

.stTextArea textarea {{
    background: rgba(255,255,255,0.05) !important;
    border-radius: 14px !important;
    border: 1px solid rgba(160,190,255,0.3) !important;
    color: #eaf2ff !important;
    font-size: 16px !important;
}}

.stTextArea textarea:focus {{
    border-color: #7aa2ff !important;
    box-shadow: 0 0 0 3px rgba(122,162,255,0.25) !important;
}}

.stButton>button {{
    background: linear-gradient(135deg,#4f8cff,#2c5cff);
    color: white !important;
    font-size: 18px !important;
    border-radius: 999px !important;
    padding: 14px !important;
    font-weight: 600 !important;
    width: 100%;
    border: none !important;
}}

.stButton>button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 12px 32px rgba(79,140,255,0.35);
}}

.pred-card {{
    margin-top: 32px;
    background: rgba(12,28,68,0.9);
    padding: 34px;
    border-radius: 24px;
    text-align: center;
    border: 1px solid rgba(160,190,255,0.18);
    backdrop-filter: blur(16px);
    animation: fadeUp 0.5s ease-out;
}}

@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.footer {{
    margin-top: 50px;
    text-align: center;
    font-size: 14px;
    opacity: 0.45;
    padding-bottom: 16px;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}
</style>

<script>
(function() {{
    function update() {{
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
        const progress = Math.min(Math.max(scrollTop / maxScroll, 0), 1);
        document.documentElement.style.setProperty('--scroll-opacity', progress);
    }}
    window.addEventListener("scroll", update);
    setTimeout(update, 200);
}})();
</script>
""", unsafe_allow_html=True)

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf.pkl"))
    label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
    return model, tfidf, label_encoder

# ===================== CLEAN TEXT =====================
def clean_text(text):
    text = text.lower()
    return re.sub(r"[^a-z\s]", "", text)

# ===================== EXAMPLES =====================
EXAMPLES = {
    "Rock": "Screaming guitars and pounding drums, lost in the noise of the night",
    "Hip-Hop": "Flowing with rhythm, dropping bars, stories from the street",
    "Pop": "Dancing under city lights, hearts colliding, feeling alive",
    "Country": "Dirt roads, old trucks, sunsets and memories",
    "Blues": "Lonely nights, broken hearts, slow melodies"
}

# ===================== APP =====================
def main():
    model, tfidf, label_encoder = load_model()

    st.markdown("""
    <div class="hero">
        <div class="hero-title">Genre Guesser</div>
        <div class="hero-sub">
            Type a few lines of lyrics.  
            Let the model gently figure out where the song belongs.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        cols = st.columns(len(EXAMPLES))
        for i, (g, text) in enumerate(EXAMPLES.items()):
            with cols[i]:
                if st.button(g):
                    st.session_state.sample = text
                    st.rerun()

        lyrics = st.text_area(
            "Lyrics",
            value=st.session_state.pop("sample", ""),
            placeholder="Write a few lines here…",
            height=200
        )

        if st.button("Predict Genre"):
            if lyrics.strip() == "":
                st.warning("Please enter some lyrics first.")
            else:
                cleaned = clean_text(lyrics)
                X = tfidf.transform([cleaned])
                pred = model.predict(X)[0]
                genre = label_encoder.inverse_transform([pred])[0]
                confidence = max(model.predict_proba(X)[0]) * 100

                st.markdown(f"""
                <div class="pred-card">
                    <h3>Predicted Genre</h3>
                    <h2>{genre.upper()}</h2>
                    <p>Confidence: {confidence:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<div style='height:50vh'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        Built with machine learning • music • curiosity
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
