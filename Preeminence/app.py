import streamlit as st
import joblib
import re

# Load saved models
@st.cache_resource
def load_models():
    model = joblib.load('model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, tfidf, label_encoder

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    return re.sub(r"[^a-z\s]", "", text)

# Load models
model, tfidf, label_encoder = load_models()

# Page config
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
    }
    
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 40px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 3.5rem;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 30px;
    }
    
    .stTextArea textarea {
        border: 2px solid #e0e0e0;
        border-radius: 15px;
        padding: 15px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 20px;
        font-weight: 600;
        padding: 15px 40px;
        border-radius: 50px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }
    
    .genre-result {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin: 20px 0;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .confidence-card {
        background: white;
        border-left: 5px solid #667eea;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .confidence-card:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }
    
    .genre-name {
        font-weight: 600;
        font-size: 1.1rem;
        color: #333;
    }
    
    .confidence-percent {
        color: #667eea;
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 20px 0;
    }
    
    .footer {
        text-align: center;
        color: #999;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 2px solid #e0e0e0;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .example-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 2px dashed #667eea;
    }
    
    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎵 About")
    st.markdown("""
    This AI-powered tool analyzes song lyrics and predicts their musical genre using machine learning.
    
    ### How it works:
    1. **Enter Lyrics** - Paste song lyrics
    2. **AI Analysis** - TF-IDF + SGD Classifier
    3. **Get Results** - Genre prediction with confidence
    
    ### Model Info:
    - **Algorithm**: SGD Classifier
    - **Features**: TF-IDF (8000 features)
    - **Accuracy**: ~59%
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Quick Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="stat-card"><div class="stat-number">10+</div><div class="stat-label">Genres</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="stat-card"><div class="stat-number">8K</div><div class="stat-label">Features</div></div>', unsafe_allow_html=True)

# Main content
st.markdown("<h1>🎵 Music Genre Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Discover the genre of any song using AI-powered lyrics analysis</p>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📝 Examples", "ℹ️ Info"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Enter Song Lyrics")
        lyrics_input = st.text_area(
            "Paste or type lyrics here:",
            height=300,
            placeholder="🎤 Enter your song lyrics here...\n\nExample:\nI see trees of green, red roses too\nI see them bloom for me and you\nAnd I think to myself\nWhat a wonderful world...",
            label_visibility="collapsed"
        )
        
        if st.button("🎯 Predict Genre", use_container_width=True):
            if lyrics_input.strip() == "":
                st.warning("⚠️ Please enter some lyrics first!")
            else:
                with st.spinner("🔮 Analyzing lyrics..."):
                    # Clean the input
                    cleaned_lyrics = clean_text(lyrics_input)
                    
                    # Transform using TF-IDF
                    lyrics_tfidf = tfidf.transform([cleaned_lyrics])
                    
                    # Predict
                    prediction_encoded = model.predict(lyrics_tfidf)
                    predicted_genre = label_encoder.inverse_transform(prediction_encoded)[0]
                    
                    # Display main result
                    st.markdown(f'<div class="genre-result">🎸 {predicted_genre}</div>', unsafe_allow_html=True)
                    
                    # Show confidence scores if available
                    if hasattr(model, 'decision_function'):
                        scores = model.decision_function(lyrics_tfidf)[0]
                        top_3_idx = scores.argsort()[-3:][::-1]
                        
                        st.markdown("### 📊 Top 3 Predictions")
                        for i, idx in enumerate(top_3_idx):
                            genre_name = label_encoder.inverse_transform([idx])[0]
                            score = scores[idx]
                            # Normalize score to percentage (rough approximation)
                            confidence = min(100, max(0, (score + 2) * 25))
                            
                            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                            st.markdown(f"""
                                <div class="confidence-card">
                                    <span style="font-size: 1.5rem;">{medal}</span>
                                    <span class="genre-name">{genre_name}</span>
                                    <span style="float: right;" class="confidence-percent">{confidence:.1f}%</span>
                                </div>
                            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 💡 Tips")
        st.markdown("""
        <div class="info-box">
        <strong>For best results:</strong><br><br>
        ✅ Use at least 4-5 lines<br>
        ✅ Include chorus or hook<br>
        ✅ Paste clean lyrics<br>
        ✅ Avoid special characters<br>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎨 Supported Genres")
        genres = label_encoder.classes_
        for genre in genres[:5]:
            st.markdown(f"🎵 {genre}")
        if len(genres) > 5:
            with st.expander("Show more..."):
                for genre in genres[5:]:
                    st.markdown(f"🎵 {genre}")

with tab2:
    st.markdown("### 📝 Example Lyrics to Try")
    
    examples = {
        "Rock": "I'm on the highway to hell\nHighway to hell\nNo stop signs, speed limit\nNobody's gonna slow me down",
        "Pop": "Cause baby you're a firework\nCome on show them what you're worth\nMake them go oh oh oh\nAs you shoot across the sky",
        "Hip Hop": "Started from the bottom now we here\nStarted from the bottom now my whole team here\nStarted from the bottom now we here",
        "Country": "Take me home country roads\nTo the place I belong\nWest Virginia mountain mama\nTake me home country roads"
    }
    
    for genre, lyrics in examples.items():
        with st.expander(f"🎵 {genre} Example"):
            st.markdown(f'<div class="example-box">{lyrics}</div>', unsafe_allow_html=True)
            if st.button(f"Try this example", key=genre):
                st.session_state.example_lyrics = lyrics
                st.rerun()

with tab3:
    st.markdown("### ℹ️ About This Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🤖 Machine Learning Model
        - **Algorithm**: Stochastic Gradient Descent (SGD) Classifier
        - **Features**: TF-IDF Vectorization (8000 features)
        - **Training**: Cross-validated on large lyrics dataset
        
        #### 🎯 How it Works
        1. Text preprocessing (lowercase, remove special chars)
        2. TF-IDF feature extraction
        3. SGD classification
        4. Genre prediction with confidence scores
        """)
    
    with col2:
        st.markdown("""
        #### 📊 Model Performance
        - Validation Accuracy: ~59%
        - Cross-Validation: ~52%
        - Multi-class classification
        
        #### 🔧 Technical Stack
        - Python 3.x
        - Scikit-learn
        - Streamlit
        - Pandas
        """)
    
    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <strong>Note:</strong> The model's predictions are based on textual patterns in lyrics. 
    Accuracy may vary depending on lyric quality, genre boundaries, and unique writing styles.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using Streamlit and Scikit-learn</p>
        <p>🎵 Music Genre Classifier v1.0 | 2024</p>
    </div>
""", unsafe_allow_html=True)