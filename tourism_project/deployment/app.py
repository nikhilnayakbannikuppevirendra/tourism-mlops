# app.py
# Purpose: Streamlit frontend for Wellness Tourism Package
#          Purchase Prediction – deployed on Hugging Face Spaces


import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ── Page Configuration 
st.set_page_config(
    page_title="Tourism Package Predictor",
    page_icon="",
    layout="centered"
)

# ── Load Model from Hugging Face Model Hub 
@st.cache_resource
def load_model():
    HF_USERNAME = "nikhilnayakbv"
    model_path = hf_hub_download(
        repo_id=f"{HF_USERNAME}/tourism-wellness-model",
        filename="best_tourism_model_v1.joblib"
    )
    return joblib.load(model_path)

model = load_model()

# ── App Header 
st.title(" Wellness Tourism Package Purchase Predictor")
st.markdown("""
This tool helps **Visit with Us** marketing teams identify customers who are likely to purchase the
new **Wellness Tourism Package** before reaching out, enabling targeted and efficient campaigns.

> **Instructions:** Fill in the customer details below and click **Predict** to see the likelihood.
""")
st.divider()

# ── Input Form 
st.subheader("👤 Customer Details")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=35)
    CityTier = st.selectbox("City Tier", [1, 2, 3],
                            help="1 = Major Metro, 2 = Tier-2 City, 3 = Smaller City")
    Occupation = st.selectbox("Occupation",
                              ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    Designation = st.selectbox("Designation",
                               ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    MonthlyIncome = st.number_input("Monthly Income (₹)", min_value=1000,
                                    max_value=100000, value=22000, step=500)

with col2:
    TypeofContact = st.selectbox("Type of Contact",
                                 ["Self Enquiry", "Company Invited"])
    ProductPitched = st.selectbox("Product Pitched",
                                  ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1,
                                             max_value=10, value=2)
    NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0.0,
                                        max_value=10.0, value=3.0, step=1.0)
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0,
                                      max_value=120.0, value=15.0, step=1.0)
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [3.0, 4.0, 5.0])

st.divider()
st.subheader("🧳 Travel Preferences")

col3, col4 = st.columns(2)
with col3:
    NumberOfTrips = st.number_input("Average Annual Trips", min_value=0.0,
                                    max_value=20.0, value=2.0, step=1.0)
    NumberOfChildrenVisiting = st.number_input("Number of Children (under 5) Visiting",
                                               min_value=0.0, max_value=5.0, value=1.0)
with col4:
    Passport = st.selectbox("Holds a Passport?", ["Yes", "No"])
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])

# ── Build Input DataFrame 
input_data = pd.DataFrame([{
    "Age":                       Age,
    "CityTier":                  CityTier,
    "DurationOfPitch":           DurationOfPitch,
    "NumberOfPersonVisiting":    NumberOfPersonVisiting,
    "NumberOfFollowups":         NumberOfFollowups,
    "PreferredPropertyStar":     PreferredPropertyStar,
    "NumberOfTrips":             NumberOfTrips,
    "Passport":                  1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore":    PitchSatisfactionScore,
    "OwnCar":                    1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting":  NumberOfChildrenVisiting,
    "MonthlyIncome":             MonthlyIncome,
    "TypeofContact":             TypeofContact,
    "Occupation":                Occupation,
    "Gender":                    Gender,
    "ProductPitched":            ProductPitched,
    "MaritalStatus":             MaritalStatus,
    "Designation":               Designation,
}])

# ── Predict 
CLASSIFICATION_THRESHOLD = 0.45

st.divider()
if st.button("🔍 Predict Purchase Likelihood", type="primary", use_container_width=True):
    proba = model.predict_proba(input_data)[0, 1]
    prediction = int(proba >= CLASSIFICATION_THRESHOLD)

    if prediction == 1:
        st.success(f" **This customer is LIKELY to purchase the Wellness Tourism Package.**")
        st.metric("Purchase Probability", f"{proba * 100:.1f}%")
        st.info(" Recommendation: Prioritise this customer in the marketing campaign.")
    else:
        st.warning(f" **This customer is UNLIKELY to purchase the Wellness Tourism Package.**")
        st.metric("Purchase Probability", f"{proba * 100:.1f}%")
        st.info(" Recommendation: Consider lower-priority outreach or alternative packages.")

st.divider()
st.caption("Powered by XGBoost + Streamlit | MLOps Pipeline – Visit with Us")
