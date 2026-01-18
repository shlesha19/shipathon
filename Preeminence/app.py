import streamlit as st
import pandas as pd
import re
import joblib
import os
import base64
from pathlib import Path

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Genre Guesser 🎪",
    page_icon="🎧",
    layout="wide"
)

# ===================== HELPER FUNCTIONS =====================
def get_base64_image(image_path):
    """Convert image to base64 for embedding in CSS"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# Get base64 encoded images
frame1_b64 = get_base64_image("frame1.png")
frame2_b64 = get_base64_image("frame2.png")

# ===================== CSS (Festival aesthetic with scroll transition) =====================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800;900&display=swap');

html, body, [class*="css"] {{
    font-family: 'Poppins', sans-serif !important;
}}

/* Background with scroll transition effect */
.stApp {{
    background: #000000 !important;
    color: #eee !important;
}}

/* Fixed background layers */
.stApp::before {{
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100vh;
    background-image: url('data:image/png;base64,{frame1_b64 if frame1_b64 else ""}');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    z-index: 0;
    pointer-events: none;
}}

.stApp::after {{
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100vh;
    background-image: url('data:image/png;base64,{frame2_b64 if frame2_b64 else ""}');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    opacity: 0;
    z-index: 0;
    pointer-events: none;
    transition: opacity 0.3s ease;
}}

/* Fallback gradient if images don't load */
{f'''
.stApp {{
    background: radial-gradient(circle at top,#1e0036,#12001e,#09000e 65%) !important;
}}
''' if not frame1_b64 else ''}

/* Hero */
.hero {{
    padding: 90px 20px;
    text-align: center;
    color: white;
}}

.hero-title {{
    font-size: 60px;
    font-weight: 900;
    letter-spacing: 2px;
    text-transform: uppercase;
    background: linear-gradient(90deg,#ff00ea,#ffcb3c,#00ffe7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 10px;
    text-shadow: 0 0 30px rgba(255,0,234,0.5);
}}

.hero-sub {{
    font-size: 20px;
    margin-top: 10px;
    opacity: 0.85;
    color: white;
    text-shadow: 0 0 20px rgba(255,203,60,0.3);
}}

/* Input container */
.input-box {{
    background: rgba(0,0,0,0.6);
    padding: 26px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15);
    margin-bottom: 8px;
}}

/* Text Area */
.stTextArea textarea {{
    background: rgba(0,0,0,0.5) !important;
    border: 2px solid rgba(255,255,255,0.2) !important;
    border-radius: 12px !important;
    color: white !important;
    font-size: 16px !important;
}}

.stTextArea textarea::placeholder {{
    color: rgba(255,255,255,0.5) !important;
}}

.stTextArea textarea:focus {{
    border-color: #ff00ea !important;
    box-shadow: 0 0 15px rgba(255,0,234,0.3) !important;
}}

.stTextArea label {{
    color: white !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}}

/* Button */
.stButton>button {{
    background: linear-gradient(135deg,#ff3caa,#8b41ff) !important;
    color: white !important;
    font-size: 20px !important;
    border-radius: 40px !important;
    padding: 14px 32px !important;
    font-weight: 700 !important;
    border: none !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
}}

.stButton>button:hover {{
    transform: scale(1.03) !important;
    box-shadow: 0 0 22px rgba(255,60,186,0.5) !important;
}}

/* Prediction Card */
.pred-card {{
    margin-top: 24px;
    background: rgba(25,16,53,0.9);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 30px;
    border-radius: 20px;
    text-align: center;
    color: white;
    animation: slideUp 0.5s ease-out;
    backdrop-filter: blur(15px);
}}

@keyframes slideUp {{
    from {{
        opacity: 0;
        transform: translateY(30px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

.pred-card h3 {{
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 15px;
    color: #ffcb3c;
}}

.pred-card h2 {{
    font-size: 44px;
    font-weight: 900;
    background: linear-gradient(90deg,#ff00ea,#ffcb3c,#00ffe7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 20px 0;
}}

.pred-card p {{
    opacity: 0.8;
    font-size: 18px;
}}

/* Warning/Info */
.stWarning, .stInfo {{
    background: rgba(255,203,60,0.25) !important;
    border: 1px solid rgba(255,203,60,0.4) !important;
    color: #ffcb3c !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px);
}}

/* Spinner */
.stSpinner > div {{
    border-top-color: #ff00ea !important;
}}

/* Footer */
.footer {{
    margin-top: 40px;
    text-align: center;
    opacity: 0.45;
    font-size: 14px;
    padding-bottom: 10px;
    color: white;
}}

/* Hide Streamlit Branding */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* Center text style */
.center-text {{
    text-align: center;
    color: white;
    margin-bottom: 20px;
    font-size: 16px;
    font-weight: 600;
}}
</style>

<!-- Scroll Effect JavaScript -->
<script>
(function() {{
    let ticking = false;
    
    function updateScrollEffect() {{
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const windowHeight = window.innerHeight;
        const docHeight = Math.max(
            document.body.scrollHeight,
            document.documentElement.scrollHeight
        );
        
        const maxScroll = Math.max(docHeight - windowHeight, 1);
        let progress = Math.min(Math.max(scrollTop / maxScroll, 0), 1);
        
        // Apply opacity to ::after pseudo-element via CSS variable
        document.documentElement.style.setProperty('--scroll-opacity', progress);
        
        // Also update stApp::after directly if possible
        const style = document.createElement('style');
        style.innerHTML = `.stApp::after {{ opacity: ${{progress}} !important; }}`;
        const oldStyle = document.getElementById('scroll-style');
        if (oldStyle) oldStyle.remove();
        style.id = 'scroll-style';
        document.head.appendChild(style);
        
        ticking = false;
    }}
    
    function requestTick() {{
        if (!ticking) {{
            window.requestAnimationFrame(updateScrollEffect);
            ticking = true;
        }}
    }}
    
    window.addEventListener('scroll', requestTick);
    window.addEventListener('resize', requestTick);
    
    // Initial call
    setTimeout(updateScrollEffect, 100);
}})();
</script>
""", unsafe_allow_html=True)

# ===================== LOAD MODEL =====================
@st.cache_resource
def load_model():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
        tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf.pkl"))
        label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
        
        return model, tfidf, label_encoder
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

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
        st.markdown('<div class="center-text">🎼 Try Example Lyrics:</div>', unsafe_allow_html=True)
        
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
            height=200,
            key="lyrics_input"
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
    
    # Add spacing for scroll effect
    st.markdown("<div style='height: 50vh;'></div>", unsafe_allow_html=True)
    
    # ===================== FOOTER =====================
    st.markdown("""
    <div class="footer">
    Made with ML • Music • Chaos
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
