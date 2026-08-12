# ==========================================================
# Flight Ticket Price Prediction Web Application
# Import Required Libraries
# ==========================================================

# Standard Library
import os
import warnings

# Data handling
import numpy as np
import pandas as pd

# Model Loading
import joblib

# streamlit
import streamlit as st

# Ignore Warnings
warnings.filterwarnings("ignore")

# ==========================================================
# Streamlit Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Flight Ticket Price Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a Bug": None,
        "About": """
        ## Flight Ticket Price Prediction

        **Machine Learning Web Application**

        Developed using:
        - Python
        - Streamlit
        - Scikit-Learn

        Model predicts airline ticket prices using historical flight data.
        """
    }
)

# ==========================================================
# Section 3 : Custom CSS Styling
# ==========================================================

st.markdown("""
<style>

/* ==========================================================
   Import Google Font
========================================================== */

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family: 'Poppins', sans-serif;
}


/* ==========================================================
   Main Background
========================================================== */

.stApp{
    background: linear-gradient(
        135deg,
        #e2e8f0 0%,
        #bfdbfe 50%,
        #93c5fd 100%
    );
}


/* ==========================================================
   Main Title
========================================================== */

.main-title{
    text-align:center;
    font-size:42px;
    font-weight:700;
    color:#0f172a !important;
    margin-top:10px;
    margin-bottom:5px;
}


.subtitle{
    text-align:center;
    font-size:18px;
    color:#334155 !important;
    margin-bottom:35px;
}


/* Force readable colors for any heading/paragraph rendered
   inside a glass-card, since Streamlit's own theme CSS
   (which targets raw h1/h3/p tags) is more specific than a
   single class selector and was overriding our colors */
.glass-card h1,
.glass-card h2,
.glass-card h3,
.glass-card h4,
.glass-card p{
    color:#0f172a !important;
}


/* ==========================================================
   Glass Card
========================================================== */

.glass-card{

    background:rgba(255,255,255,0.94);

    backdrop-filter:blur(12px);

    border-radius:18px;

    padding:25px;

    box-shadow:0 8px 30px rgba(15,23,42,0.15);

    border:1px solid rgba(15,23,42,0.08);

    margin-bottom:20px;
}


/* ==========================================================
   Prediction Card
========================================================== */

.prediction-card{

    background:linear-gradient(
        135deg,
        #2563eb,
        #1d4ed8
    );

    color:white;

    border-radius:18px;

    padding:30px;

    text-align:center;

    box-shadow:0 10px 30px rgba(37,99,235,.35);

    margin-top:20px;
}


.prediction-price{

    font-size:42px;

    font-weight:bold;

    margin-top:10px;
}


/* ==========================================================
   Success / Info Alert Boxes (st.success, st.info)
   Streamlit's default alert text color is made for a dark
   theme and looks washed-out/low-contrast on our light
   background, so we override it explicitly.
========================================================== */

div[data-testid="stAlert"]{

    background:#dcfce7 !important;

    border:1px solid #16a34a;

    border-radius:12px;
}

div[data-testid="stAlert"] p{

    color:#14532d !important;

    font-weight:600;
}


/* ==========================================================
   Sidebar
========================================================== */

section[data-testid="stSidebar"]{

    background:#0f172a;

}


section[data-testid="stSidebar"] *{

    color:white;

}


/* ==========================================================
   Buttons
========================================================== */

div.stButton > button{

    width:100%;

    background:linear-gradient(
        90deg,
        #2563eb,
        #1d4ed8
    );

    color:white;

    border:none;

    border-radius:12px;

    height:55px;

    font-size:18px;

    font-weight:600;

    transition:0.3s;

}


div.stButton > button:hover{

    transform:translateY(-2px);

    box-shadow:0 8px 20px rgba(37,99,235,.35);

}


/* ==========================================================
   Input Boxes
========================================================== */

div[data-baseweb="select"]{

    border-radius:12px;

}


input{

    border-radius:12px !important;

}


/* ==========================================================
   Footer
========================================================== */

.footer{

    text-align:center;

    color:#1e293b !important;

    margin-top:40px;

    font-size:14px;

    padding:20px;

    background:rgba(255,255,255,0.85);

    border-radius:16px;

}


.footer h3{

    color:#0f172a !important;

    margin-bottom:10px;

}


.footer p{

    color:#334155 !important;

    margin:4px 0;

}

</style>
""", unsafe_allow_html=True)

# ==========================================
# Load Model & Preprocessor
# ==========================================

# Artifact Paths
MODEL_PATH = "artifacts/best_model.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessor.pkl"


@st.cache_resource(show_spinner="Loading Machine Learning Model...")
def load_artifacts():
    """
    Load the trained model and preprocessing pipeline.

    Returns
    -------
    tuple
        (preprocessor, model)
    """

    # Check if artifact files exist
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.stop()

    if not os.path.exists(PREPROCESSOR_PATH):
        st.error(f"❌ Preprocessor file not found: {PREPROCESSOR_PATH}")
        st.stop()

    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)

        return preprocessor, model

    except Exception as e:
        st.error(f"❌ Error while loading artifacts:\n\n{e}")
        st.stop()


# Load Artifacts
preprocessor, best_model = load_artifacts()

# ==============================================
# Flight Price Prediction Function
# ==============================================

def predict_flight_price(input_data):
    """
    Predict the flight ticket price using the trained model.

    Parameters
    ----------
    input_data : dict
        Dictionary containing flight details.

    Returns
    -------
    float
        Predicted flight ticket price.
    """

    try:

        # Convert dictionary into DataFrame
        input_df = pd.DataFrame([input_data])

        # Apply preprocessing
        processed_data = preprocessor.transform(input_df)

        # Generate prediction
        prediction = best_model.predict(processed_data)

        # Return prediction
        return round(float(prediction[0]), 2)

    except Exception as e:

        st.error("Prediction Failed!")
        st.exception(e)

        return None

# =========================================
# Professional Sidebar
# =========================================

with st.sidebar:

    # ------------------------------------------------------
    # Project Logo / Title
    # ------------------------------------------------------
    st.markdown("""
    <h1 style='text-align:center; color:#4F8BF9;'>
        ✈️ Flight Price
    </h1>

    <h4 style='text-align:center; color:white;'>
        Prediction System
    </h4>

    <hr>
    """, unsafe_allow_html=True)

    # ------------------------------------------------------
    # Model Status
    # ------------------------------------------------------
    st.subheader("🤖 Model Status")

    st.success("Model Loaded Successfully")

    st.info("Prediction System Ready")

    st.divider()

    # ------------------------------------------------------
    # Technologies Used
    # ------------------------------------------------------
    st.subheader("🛠️ Technologies")

    st.markdown("""
    - Python
    - Streamlit
    - Scikit-Learn
    - Pandas
    - NumPy
    - Joblib
    """)

    st.divider()

    # ------------------------------------------------------
    # Model Information
    # ------------------------------------------------------
    st.subheader("📊 Model Information")

    st.write("**Algorithm:** Random Forest Regressor")

    st.write("**Prediction Type:** Regression")

    st.write("**Dataset:** Flight Ticket Price")

    st.divider()

    # ------------------------------------------------------
    # Instructions
    # ------------------------------------------------------
    st.subheader("📖 How to Use")

    st.markdown("""
    1. Select Airline
    2. Select Source City
    3. Select Destination
    4. Enter Duration
    5. Enter Days Left
    6. Click **Predict**
    """)

    st.divider()

    # ------------------------------------------------------
    # Version
    # ------------------------------------------------------
    st.subheader("ℹ️ Version")

    st.write("Version : 1.0.0")

    st.write("Status : Production Ready")

    st.divider()

    # ------------------------------------------------------
    # Developer
    # ------------------------------------------------------
    st.markdown("""
    <div style="text-align:center;">

    👨‍💻 <b>Developed by</b>

    <br>

    <b>Nitin Maurya</b>

    <br><br>

    Machine Learning Project

    </div>
    """, unsafe_allow_html=True)

# ==============================================
# Main Dashboard
# ==============================================

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="glass-card">

<h1 class="main-title">
✈️ Flight Ticket Price Prediction
</h1>

<p class="subtitle">
Predict airline ticket prices instantly using our Machine Learning model.
</p>

</div>
""", unsafe_allow_html=True)


# -----------------------------
# Flight Details Form
# -----------------------------
st.markdown("""
<div class="glass-card">
<h3>📝 Enter Flight Details</h3>
</div>
""", unsafe_allow_html=True)


# Create Two Columns
col1, col2 = st.columns(2)


# ==========================================================
# Left Column
# ==========================================================

with col1:

    airline = st.selectbox(
        "✈ Airline",
        [
            "AirAsia",
            "Air India",
            "GO FIRST",
            "Indigo",
            "SpiceJet",
            "Vistara"
        ]
    )

    source_city = st.selectbox(
        "📍 Source City",
        [
            "Delhi",
            "Mumbai",
            "Bangalore",
            "Kolkata",
            "Hyderabad",
            "Chennai"
        ]
    )

    departure_time = st.selectbox(
        "🕒 Departure Time",
        [
            "Early Morning",
            "Morning",
            "Afternoon",
            "Evening",
            "Night",
            "Late Night"
        ]
    )

    stops = st.selectbox(
        "🛑 Stops",
        [
            "zero",
            "one",
            "two_or_more"
        ]
    )



# ==========================================================
# Right Column
# ==========================================================

with col2:

    arrival_time = st.selectbox(
        "🛬 Arrival Time",
        [
            "Early Morning",
            "Morning",
            "Afternoon",
            "Evening",
            "Night",
            "Late Night"
        ]
    )

    destination_city = st.selectbox(
        "🎯 Destination City",
        [
            "Delhi",
            "Mumbai",
            "Bangalore",
            "Kolkata",
            "Hyderabad",
            "Chennai"
        ]
    )

    travel_class = st.selectbox(
        "💺 Travel Class",
        [
            "Economy",
            "Business"
        ]
    )

    duration = st.number_input(
        "⏱ Duration (Hours)",
        min_value=0.5,
        max_value=50.0,
        value=2.0,
        step=0.1
    )

    days_left = st.number_input(
        "📅 Days Left Before Journey",
        min_value=1,
        max_value=365,
        value=30,
        step=1
    )


# ==========================================================
# Input Dictionary
# ==========================================================

flight_details = {

    "airline": airline,

    "source_city": source_city,

    "departure_time": departure_time,

    "stops": stops,

    "arrival_time": arrival_time,

    "destination_city": destination_city,

    "class": travel_class,

    "duration": duration,

    "days_left": days_left
}

# =========================================
# Prediction Logic
# =========================================

# Initialize Session State
if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ----------------------------------------------------------
# Predict Button
# ----------------------------------------------------------

predict_button = st.button(
    "🔮 Predict Ticket Price",
    use_container_width=True,
    type="primary"
)


# ----------------------------------------------------------
# Prediction Process
# ----------------------------------------------------------

if predict_button:

    # Basic Input Validation
    if duration <= 0:
        st.error("❌ Duration must be greater than 0.")
        st.stop()

    if days_left < 1:
        st.error("❌ Days Left must be at least 1.")
        st.stop()

    try:

        with st.spinner("🤖 AI Model is predicting ticket price..."):

            prediction = predict_flight_price(flight_details)

            st.session_state.prediction = prediction

        st.success("✅ Prediction Generated Successfully!")

    except Exception as e:

        st.session_state.prediction = None

        st.error("❌ Unable to generate prediction.")

        st.exception(e)

# ==========================================================
# Prediction Result Dashboard
# ==========================================================

# Display results only after prediction
if st.session_state.prediction is not None:

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------
    # Prediction Card
    # ------------------------------------------------------
    st.markdown(f"""
<div class="prediction-card">
<h2>💰 Predicted Flight Ticket Price</h2>
<div class="prediction-price">₹ {st.session_state.prediction:,.2f}</div>
<p>Estimated ticket price generated by the trained Machine Learning model.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ------------------------------------------------------
    # Metrics
    # ------------------------------------------------------
    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.metric(
            label="🤖 Model",
            value="Random Forest"
        )

    with metric2:
        st.metric(
            label="📈 Prediction",
            value="Success"
        )

    with metric3:
        st.metric(
            label="⚡ Status",
            value="Ready"
        )

    st.markdown("---")

    # ------------------------------------------------------
    # Flight Summary
    # ------------------------------------------------------
    st.subheader("✈️ Flight Summary")

    summary_df = pd.DataFrame({

        "Feature": [
            "Airline",
            "Source City",
            "Destination City",
            "Departure Time",
            "Arrival Time",
            "Stops",
            "Travel Class",
            "Duration (Hours)",
            "Days Left"
        ],

        "Value": [

            airline,

            source_city,

            destination_city,

            departure_time,

            arrival_time,

            stops,

            travel_class,

            duration,

            days_left
        ]

    })

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    # ------------------------------------------------------
    # Success Message
    # ------------------------------------------------------
    st.success(
        "Prediction completed successfully. "
        "Review the estimated ticket price and flight details above."
    )

# =========================================
# Professional Footer
# =========================================

st.markdown("---")

col1, col2, col3 = st.columns(3)

# ==========================================================
# About Project
# ==========================================================

with col1:

    st.markdown("""
    ### ✈️ About Project

    Flight Ticket Price Prediction is a Machine Learning
    application that estimates airline ticket prices based
    on flight details entered by the user.

    The prediction is generated using a trained
    Scikit-Learn regression model.
    """)


# ==========================================================
# Technologies Used
# ==========================================================

with col2:

    st.markdown("""
    ### 🛠 Technologies

    - Python
    - Streamlit
    - Scikit-Learn
    - Pandas
    - NumPy
    - Joblib
    """)


# ==========================================================
# Project Information
# ==========================================================

with col3:

    st.markdown("""
    ### 📌 Project Details

    **Version:** 1.0.0

    **Status:** Production Ready

    **Model:** Random Forest Regressor

    **Type:** Regression
    """)


st.markdown("---")


# ==========================================================
# Footer
# ==========================================================

st.markdown("""
<div class="footer">

<h3>👨‍💻 Developed by Nitin Maurya</h3>

<p>
Flight Ticket Price Prediction System
</p>

<p>
Built using Python • Streamlit • Scikit-Learn
</p>

<p>
© 2026 All Rights Reserved
</p>

</div>
""", unsafe_allow_html=True)
