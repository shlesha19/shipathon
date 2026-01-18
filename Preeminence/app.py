import streamlit as st
import pandas as pd
import re
import joblib
import os
from pathlib import Path

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="VibeCheck.AI 🎧✨",
    page_icon="🎶",
    layout="centered"
)

# ===================== CSS HYPER AURA =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');

/* Global Styles */
.stApp {
    background: linear-gradient(135deg, #fface4, #a7d5ff, #ffd4b8, #c3ffd4);
    background-size: 400% 400%;
    animation: gradientMove 18s ease infinite;
    font-family: 'Inter', sans-serif;
}

@keyframes gradientMove {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Title Styles */
.title {
    font-size: 54px;
    font-weight: 900;
    text-align: center;
    letter-spacing: -1px;
    color: #111;
    text-shadow: 0px 3px 8px rgba(0,0,0,0.15);
    margin-bottom: 0px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-top: -8px;
    color: #222;
    margin-bottom: 30px;
}

/* Bubble Containers */
.bubble {
    background: rgba(255,255,255,0.65);
    border-radius: 18px;
    padding: 18px;
    margin: 12px 0px;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.35);
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
}

/* Result Card */
.result {
    background: rgba(255,255,255,0.85);
    padding: 24px;
    border-radius: 18px;
    text-align: center;
    backdrop-filter: blur(15px);
    border: 2px solid rgba(255,255,255,0.5);
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    margin: 20px 0px;
}

.genre-display {
    font-size: 48px;
    font-weight: 900;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 15px 0px;
    text-shadow: 0px 2px 5px rgba(99,102,241,0.2);
}

.confidence-text {
    font-size: 20px;
    color: #555;
    font-weight: 600;
    margin-top: 10px;
}

/* Progress Bar */
.confidence-bar {
    background: rgba(200,200,200,0.3);
    height: 24px;
    border-radius: 12px;
    overflow: hidden;
    margin: 15px 0px;
}

.confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    border-radius: 12px;
    transition: width 0.8s ease-out;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 12px;
    color: white;
    font-weight: 700;
    font-size: 14px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 28px;
    font-weight: 700;
    font-size: 16px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3);
    width: 100%;
}

.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.4);
}

/* Text Area */
.stTextArea textarea {
    background: rgba(255,255,255,0.8);
    border: 2px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 12px;
    font-size: 15px;
    transition: all 0.3s ease;
}

.stTextArea textarea:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1);
}

/* Example Chips */
.example-chip {
    display: inline-block;
    background: rgba(255,255,255,0.7);
    padding: 8px 16px;
    border-radius: 20px;
    margin: 5px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    border: 2px solid rgba(99,102,241,0.2);
}

.example-chip:hover {
    background: #6366f1;
    color: white;
    transform: translateY(-2px);
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
def load_model():
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

# ===================== PREDICTION FUNCTION =====================
def predict_genre(lyrics, model, tfidf, label_encoder):
    """Predict genre from lyrics"""
    cleaned_lyrics = clean_text(lyrics)
    lyrics_tfidf = tfidf.transform([cleaned_lyrics])
    
    # Get prediction
    prediction_encoded = model.predict(lyrics_tfidf)
    genre = label_encoder.inverse_transform(prediction_encoded)[0]
    
    # Get confidence scores
    probabilities = model.predict_proba(lyrics_tfidf)[0]
    confidence = max(probabilities) * 100
    
    return genre, confidence

# ===================== EXAMPLE LYRICS =====================
EXAMPLES = {
    "🎸 Rock": "Screaming guitars and thundering drums, headbanging all night long, metal rules supreme, power chords echoing through the night, electric energy filling the air",
    "🎤 Hip-Hop": "Yo, dropping beats on the street, rhymes so sweet, can't be beat, rap game complete, flow so sick, lyrics quick, bass kicks thick, spitting fire on the mic",
    "🤠 Country": "Driving my truck down a dirt road, country music on the radio, simple life is all I know, boots and hat, where it's at, sweet tea and southern charm",
    "✨ Pop": "Baby you're the one for me, dancing all night feeling free, this love is meant to be, hearts collide under neon lights, we're shining bright tonight",
    "🎺 Blues": "I got the blues, feeling so down, my baby left me in this old town, walking these streets with nothing to lose, just me and my heartbreak blues, guitar crying in the night",
}

# ===================== MAIN APP =====================
def main():
    # Load model
    model, tfidf, label_encoder = load_model()
    
    # Header
    st.markdown('<div class="title">🎧 VibeCheck.AI ✨</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Discover the genre of any song using AI-powered lyrics analysis</div>', unsafe_allow_html=True)
    
    if model is None or tfidf is None or label_encoder is None:
        st.stop()
    
    # Main input area
    st.markdown('<div class="bubble">', unsafe_allow_html=True)
    lyrics_input = st.text_area(
        "Enter Song Lyrics",
        height=200,
        placeholder="Paste or type song lyrics here...\n\nExample:\nI got the blues, feeling so down\nMy baby left me in this old town\nWalking these streets with nothing to lose...",
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Example buttons
    st.markdown('<div class="bubble">', unsafe_allow_html=True)
    st.markdown("**🎼 Try Example Lyrics:**")
    
    cols = st.columns(len(EXAMPLES))
    for idx, (label, example_text) in enumerate(EXAMPLES.items()):
        with cols[idx]:
            if st.button(label, key=f"example_{idx}", use_container_width=True):
                st.session_state.selected_example = example_text
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Use selected example if available
    if 'selected_example' in st.session_state:
        lyrics_input = st.session_state.selected_example
        st.session_state.pop('selected_example')
    
    # Predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("🎯 Predict Genre", use_container_width=True)
    
    # Prediction
    if predict_button:
        if not lyrics_input.strip():
            st.warning("⚠️ Please enter some lyrics first!")
        else:
            with st.spinner("🔮 Analyzing lyrics..."):
                genre, confidence = predict_genre(lyrics_input, model, tfidf, label_encoder)
            
            # Display result
            st.markdown('<div class="result">', unsafe_allow_html=True)
            st.markdown("**Predicted Genre:**")
            st.markdown(f'<div class="genre-display">{genre}</div>', unsafe_allow_html=True)
            
            # Confidence bar
            st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {confidence}%">
                        {confidence:.1f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f'<div class="confidence-text">Confidence: {confidence:.1f}%</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Success message with emoji
            if confidence > 80:
                st.success("🎉 High confidence prediction!")
            elif confidence > 60:
                st.info("👍 Good prediction!")
            else:
                st.warning("🤔 Low confidence - the lyrics might be ambiguous")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 14px;'>"
        "Powered by TF-IDF + SGD Classifier 🤖 | Made with ❤️ and Streamlit"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

