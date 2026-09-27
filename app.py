
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Reading Preference AI",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(143, 166, 137, 0.12), transparent 30%),
        radial-gradient(circle at 90% 20%, rgba(207, 188, 153, 0.13), transparent 30%),
        #F7F5EF;
}

/* Remove default top padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}

/* Main title */
.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: #24352A;
    margin-bottom: 0.2rem;
    letter-spacing: -1px;
}

.subtitle {
    color: #4A504A; /* Darkened for better contrast */
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* Small badge */
.badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 50px;
    background: #E6EEE2;
    color: #496044;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
}

/* Section title */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.65rem;
    font-weight: 600;
    color: #2D4032;
    margin-top: 0.5rem;
    margin-bottom: 0.2rem;
}

.section-description {
    color: #5A605A; /* Darkened for better contrast */
    font-size: 0.9rem;
    margin-bottom: 1.2rem;
}

/* Cards */
.card {
    background: white; /* Changed to solid white for better contrast */
    border: 1px solid rgba(70, 88, 70, 0.10);
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 10px 30px rgba(47, 61, 48, 0.06);
    margin-bottom: 1.2rem;
}

.result-card {
    background: linear-gradient(
        135deg,
        #EFF4EB 0%,
        #F9F7F0 100%
    );
    border: 1px solid #DCE5D7;
    border-radius: 24px;
    padding: 2rem;
    margin-top: 1.5rem;
    box-shadow: 0 15px 35px rgba(52, 71, 54, 0.08);
}

.prediction-label {
    color: #74806F;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

.prediction-value {
    font-family: 'Playfair Display', serif;
    color: #2D4732;
    font-size: 2.3rem;
    font-weight: 700;
    margin-top: 0.3rem;
}

/* Probability cards */
.prob-card {
    background: white;
    border-radius: 16px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    border: 1px solid #E7EAE4;
}

.prob-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    color: #4D594D;
    font-size: 0.9rem;
    font-weight: 600;
}

.progress-bg {
    width: 100%;
    height: 8px;
    background: #E8ECE5;
    border-radius: 20px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #71866C, #9AAA91);
    border-radius: 20px;
}

/* Button */
.stButton > button {
    width: 100%;
    border: none;
    border-radius: 14px;
    padding: 0.8rem 1rem;
    background: #304A35;
    color: white;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.25s ease;
    box-shadow: 0 8px 20px rgba(48, 74, 53, 0.18);
}

.stButton > button:hover {
    background: #3E5D43;
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(48, 74, 53, 0.25);
}

/* Slider */
.stSlider > div > div > div > div {
    background-color: #71866C;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    border-radius: 12px;
    border-color: #DCE2D8;
    background-color: #FFFFFF;
}

/* Divider */
.soft-divider {
    height: 1px;
    background: #E4E7E0;
    margin: 1.5rem 0;
}

/* Footer */
.footer {
    text-align: center;
    color: #8A9088;
    font-size: 0.8rem;
    margin-top: 3rem;
}

/* Info box */
.info-box {
    background: #F0F4EC;
    border-left: 4px solid #71866C;
    padding: 1rem 1.2rem;
    border-radius: 0 12px 12px 0;
    color: #3A423A; /* Darkened for better contrast */
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
}

/* Mobile */
@media (max-width: 768px) {
    .main-title {
        font-size: 2.3rem;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD ARTIFACTS
# =========================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load("random_forest_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    feature_names = joblib.load("feature_names.pkl")
    input_metadata = joblib.load("input_metadata.pkl")

    return model, preprocessor, feature_names, input_metadata


model, preprocessor, feature_names, input_metadata = load_artifacts()


# =========================================================
# TARGET LABELS
# =========================================================

target_labels = sorted(model.classes_)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="badge">✦ AI READING ANALYTICS</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">Reading Preference</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover your reading preference through a simple AI-powered prediction.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# INTRODUCTION
# =========================================================

st.markdown("""
<div class="info-box">
    📚 Adjust the information below based on your personal characteristics
    and reading preferences. The AI model will analyze your inputs and
    estimate your reading preference.
</div>
""", unsafe_allow_html=True)


# =========================================================
# PERSONAL PROFILE
# =========================================================

st.markdown(
    '<div class="section-title">Personal Profile</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Tell us a little about yourself.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        options=input_metadata["categorical_options"]["gender"]
    )

with col2:

    age = st.slider(
        "Age",
        min_value=17,
        max_value=60,
        value=30
    )

with col3:

    education = st.selectbox(
        "Education",
        options=input_metadata["categorical_options"]["education"]
    )

col4, col5 = st.columns(2)

with col4:

    books_per_year = st.slider(
        "Books Per Year",
        min_value=0,
        max_value=30,
        value=10
    )

with col5:

    reading_frequency = st.selectbox(
        "Reading Frequency",
        options=input_metadata["categorical_options"]["reading_frequency"]
    )

st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# READING PREFERENCES
# =========================================================

st.markdown(
    '<div class="section-title">Reading Preferences</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Adjust the sliders according to your personal preferences.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:

    literary_self_rating = st.slider(
        "Literary Self Rating",
        min_value=1,
        max_value=7,
        value=4,
        help="How would you rate your own literary ability?"
    )

    story_importance = st.slider(
        "Story Importance",
        min_value=1,
        max_value=5,
        value=3
    )

    writing_style_importance = st.slider(
        "Writing Style Importance",
        min_value=1,
        max_value=5,
        value=3
    )

    deeper_meaning_preference = st.slider(
        "Deeper Meaning Preference",
        min_value=1,
        max_value=5,
        value=3
    )

with col2:

    real_events_preference = st.slider(
        "Real Events Preference",
        min_value=1,
        max_value=5,
        value=3
    )

    easy_reading_preference = st.slider(
        "Easy Reading Preference",
        min_value=1,
        max_value=5,
        value=3
    )

    intellectual_challenge = st.slider(
        "Intellectual Challenge",
        min_value=1,
        max_value=5,
        value=3
    )

st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# CREATE INPUT DATA
# =========================================================

input_data = pd.DataFrame({

    "gender": [gender],

    "age": [age],

    "education": [education],

    "books_per_year": [books_per_year],

    "reading_frequency": [reading_frequency],

    "literary_self_rating": [literary_self_rating],

    "story_importance": [story_importance],

    "writing_style_importance": [writing_style_importance],

    "deeper_meaning_preference": [deeper_meaning_preference],

    "real_events_preference": [real_events_preference],

    "easy_reading_preference": [easy_reading_preference],

    "intellectual_challenge": [intellectual_challenge]

})


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown(
    '<div style="height:10px;"></div>',
    unsafe_allow_html=True
)

if st.button("✦  Predict My Reading Preference"):

    # -----------------------------------------------------
    # PREPROCESS
    # -----------------------------------------------------

    input_processed = preprocessor.transform(input_data)

    input_processed_df = pd.DataFrame(
        input_processed,
        columns=feature_names
    )

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    probabilities = model.predict_proba(
        input_processed_df
    )[0]

    predicted_class_index = np.argmax(probabilities)

    predicted_class = target_labels[
        predicted_class_index
    ]

    # -----------------------------------------------------
    # RESULT
    # -----------------------------------------------------

    st.markdown(
        '<div class="result-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="prediction-label">'
        'YOUR PREDICTED READING PREFERENCE'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="prediction-value">'
        f'{predicted_class}'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="soft-divider"></div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title" style="font-size:1.3rem;">'
        'Prediction Confidence'
        '</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # PROBABILITY
    # -----------------------------------------------------

    for i, prob in enumerate(probabilities):

        label = target_labels[i]

        percentage = prob * 100

        st.markdown(
            f"""
            <div class="prob-card">

                <div class="prob-label">

                    <span>{label}</span>

                    <span>{percentage:.2f}%</span>

                </div>

                <div class="progress-bg">

                    <div
                        class="progress-fill"
                        style="width:{percentage}%;">
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # ADDITIONAL INFORMATION
    # -----------------------------------------------------

    st.markdown(
        f"""
        <div class="info-box" style="margin-top:20px;">

            💡 Based on the information provided, the model
            estimates <b>{predicted_class}</b> as the reading
            preference with the highest predicted probability.

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        Reading Preference AI · Powered by Random Forest
        <br>
        Designed with a natural & minimal aesthetic 🌿
    </div>
    """,
    unsafe_allow_html=True
)
