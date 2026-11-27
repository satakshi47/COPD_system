import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import date


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="LungCare | COPD Prediction",
    page_icon="🫁",
    layout="wide"
)


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .stApp {
        background-color: #f6f5ff;
        color: #19163d;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background-color: #292354;
    }

    [data-testid="stSidebar"] * {
        color: white;
    }

    h1, h2, h3 {
        color: #19163d !important;
    }

    [data-testid="stNumberInput"] label {
        color: #252247 !important;
        font-weight: 600;
    }

    [data-testid="stNumberInput"] input {
        color: #252247 !important;
        background-color: white !important;
        border: 1px solid #d7d3ef !important;
        border-radius: 9px !important;
    }

    [data-testid="stFormSubmitButton"] button {
        background-color: #5034bd !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        min-height: 44px;
        width: 100%;
    }

    [data-testid="stFormSubmitButton"] button:hover {
        background-color: #3d2797 !important;
        color: white !important;
    }

    [data-testid="stExpander"] {
        background-color: #fbfaff;
        border: 1px solid #dedaf5;
        border-radius: 10px;
    }

    .hero {
        background: linear-gradient(110deg, #e4e4ff, #f2efff);
        border: 1px solid #d8d6fa;
        border-radius: 20px;
        padding: 26px 30px;
        margin: 10px 0 20px 0;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        color: #19163d;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        font-size: 17px;
        font-weight: 600;
        color: #37316f;
        margin-bottom: 6px;
    }

    .hero-description {
        color: #4c4876;
        margin: 0;
    }

    .result-box {
        background-color: #f0edff;
        border: 1px solid #d9d1ff;
        border-radius: 15px;
        padding: 22px;
        text-align: center;
        margin: 14px 0;
    }

    .result-label {
        color: #5942bd;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1px;
    }

    .result-value {
        color: #39269c;
        font-size: 30px;
        font-weight: 800;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:10px 0 25px;">
            <div style="font-size:44px;">🫁</div>
            <h2 style="color:white; margin:0;">LungCare</h2>
            <p style="color:#e3e0ff; font-size:13px;">
                Breathe Better. Live Healthier.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("###  Home")
    st.markdown("###  COPD Prediction")
    st.markdown("###  Health Insights")
    st.markdown("###  Resources")
    st.markdown("### ⓘ About Us")

    st.markdown("---")

    st.markdown(
        """
        <div style="text-align:center; padding:20px 0; color:white;">
            ♡<br>
            <b>Your Lung Health<br>Our Priority</b>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------- LOAD MODEL AND DATA ----------------
BASE_DIR = Path(__file__).parent
model_path = BASE_DIR / "models" / "decision_tree.joblib"
training_path = BASE_DIR / "data" / "processed" / "X_train.csv"

if not model_path.exists():
    st.error("Model file not found: models/decision_tree.joblib")
    st.stop()

if not training_path.exists():
    st.error("Training data not found: data/processed/X_train.csv")
    st.stop()

model = joblib.load(model_path)

training_data = pd.read_csv(training_path)

feature_names = training_data.columns.tolist()

default_values = training_data.median(
    numeric_only=True
).to_dict()


# ---------------- FEATURE GROUPS ----------------
main_features = [
    "AGE",
    "PackHistory",
    "FEV1",
    "FEV1PRED",
    "FVC",
    "CAT"
]

optional_features = [
    feature
    for feature in feature_names
    if feature not in main_features
]


# ---------------- HERO BANNER ----------------
st.markdown(
    """
<div class="hero">
<div class="hero-title">COPD Prediction</div>
<div class="hero-subtitle">Understand your model's prediction.</div>
<p class="hero-description">Enter the available feature values below to see the predicted COPD severity class.</p>
</div>
""",
    unsafe_allow_html=True
)

st.info(
    "Educational project prototype only. This model output is not a "
    "medical diagnosis and must not be used for treatment decisions."
)


# ---------------- MAIN LAYOUT ----------------
left_col, right_col = st.columns(
    [1, 1.05],
    gap="large"
)


# ---------------- PATIENT DETAILS ----------------
with left_col:
    with st.container(border=True):

        st.subheader("Patient Details")
        st.caption("Enter the available feature values.")

        with st.form("copd_prediction_form"):

            input_values = {}

            st.markdown("**Main Features**")

            main_columns = st.columns(2)

            for i, feature in enumerate(main_features):

                with main_columns[i % 2]:

                    input_values[feature] = st.number_input(
                        feature,
                        value=float(default_values[feature]),
                        key=f"main_{feature}"
                    )

            # Optional features
            with st.expander("＋ Additional Features (Optional)"):

                st.caption(
                    "Leave fields unchanged if you do not have their "
                    "values. The app will use the training dataset median."
                )

                optional_columns = st.columns(2)

                for i, feature in enumerate(optional_features):

                    with optional_columns[i % 2]:

                        input_values[feature] = st.number_input(
                            feature,
                            value=float(default_values[feature]),
                            key=f"optional_{feature}"
                        )

            submitted = st.form_submit_button(
                "Predict COPD Severity",
                use_container_width=True
            )

        st.caption(
            f"The model uses all {len(feature_names)} features. Optional fields are "
            "prefilled with median values from the training data."
        )


# ---------------- PREDICTION RESULT ----------------
with right_col:
    with st.container(border=True):

        st.subheader("Prediction Result")

        st.caption(
            f"Prediction date: {date.today().strftime('%d %b %Y')}"
        )

        if submitted:

            input_df = pd.DataFrame([input_values])

            # Keep the same feature order as the training data
            input_df = input_df[feature_names]

            prediction = model.predict(input_df)[0]

            st.markdown(
                f"""
<div class="result-box">
<div class="result-label">MODEL PREDICTION</div>
<div class="result-value">{prediction}</div>
<div>Predicted COPD severity class</div>
</div>
""",
                unsafe_allow_html=True
            )

            st.markdown("### What this means")

            st.write(
                "This is the class predicted by the trained Decision Tree "
                "model from the values entered. It is not a percentage "
                "risk estimate or a confirmed medical diagnosis."
            )

        else:

            st.markdown(
                """
<div class="result-box">
<div class="result-label">PREDICTION RESULT</div>
<div class="result-value">—</div>
<div>Your result will appear here after prediction.</div>
</div>
""",
                unsafe_allow_html=True
            )

        st.markdown("---")

        st.markdown("### General Health Information")

        st.markdown("""
        - Avoid smoking and secondhand smoke.
        - Reduce exposure to dust and air pollution.
        - Discuss breathing symptoms with a qualified healthcare professional.
        """)
