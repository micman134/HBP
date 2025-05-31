import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from fpdf import FPDF

# Set page config
st.set_page_config(page_title="HBP Risk Prediction System", layout="wide")

# Sidebar navigation
with st.sidebar:
    st.title("Menu")
    page = st.radio("Go to", ["Predict", "Ontology", "About"])

# Load model (cached)
@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    return joblib.load('model.joblib')

# Load the model
try:
    model = load_model()
    # Add model diagnostics
    if hasattr(model, "predict_proba"):
        st.sidebar.success("Model supports probability predictions")
    else:
        st.sidebar.warning("Model only supports class predictions")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Ontology page (unchanged)
if page == "Ontology":
    st.title("Ontology For HBP Prediction System")
    # ... [keep existing ontology page code] ...

# About page (unchanged)
elif page == "About":
    st.title("About This Tool")
    # ... [keep existing about page code] ...

# Main Prediction Page
else:
    st.title("High Blood Pressure Risk Prediction Tool")
    st.write("""
    This tool predicts the risk of HBP based on patient characteristics.
    Please fill in all the fields below and click 'Predict'.
    """)

    # Initialize session state
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    # Input form (unchanged)
    with st.form("prediction_form"):
        # ... [keep existing input form code] ...
        
        submitted = st.form_submit_button("Predict HBP Risk")
        if submitted:
            st.session_state.submitted = True

    # Prediction logic
    if st.session_state.get('submitted', False):
        # Validate all fields are selected
        if (los == "Select" or ckd == "Select" or atd == "Select" or 
            gender == "Select Gender" or pregnancy == "Select" or smoking == "Select"):
            st.error("Please fill in all fields before submitting")
            st.session_state.submitted = False
        else:
            try:
                with st.spinner('Analyzing health data...'):
                    time.sleep(1)
                    
                    # Encode inputs
                    input_data = {
                        'loh': loh,
                        'gpc': gpc,
                        'age': age,
                        'bmi': bmi,
                        'gender': 1 if gender == "Female" else 0,
                        'pregnancy': 1 if pregnancy == "Yes" else 0,
                        'smoking': 1 if smoking == "Yes" else 0,
                        'pa': pa,
                        'scid': scid,
                        'alcohol': alcohol,
                        'los': ["Acute/normal stress", "Episodic acute stress", "Chronic Stress"].index(los) + 1,
                        'ckd': 1 if ckd == "Yes" else 0,
                        'atd': 1 if atd == "Yes" else 0
                    }
                    
                    features = ['loh', 'gpc', 'age', 'bmi', 'gender', 'pregnancy', 'smoking', 
                               'pa', 'scid', 'alcohol', 'los', 'ckd', 'atd']
                    input_values = [[input_data[feature] for feature in features]]
                    
                    # Get predictions with validation
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(input_values)[0]
                        
                        # Add diagnostic output
                        with st.expander("Model Diagnostics"):
                            st.write("Raw probability outputs:", proba)
                            st.write("Class prediction:", model.predict(input_values))
                            
                        # Validate probabilities
                        if np.allclose(proba.sum(), 1.0, rtol=1e-3) and (proba >= 0).all() and (proba <= 1).all():
                            prediction = 1 if proba[1] > 0.5 else 0
                        else:
                            st.error("Invalid probabilities returned by model")
                            st.write("Probabilities should sum to 1 and be between 0-1")
                            st.stop()
                    else:
                        prediction = model.predict(input_values)[0]
                        proba = [0.5, 0.5]  # Default if no probabilities available
                
                # Display results
                st.divider()
                
                # Risk Summary Section
                col1, col2 = st.columns([1, 1.5])
                
                with col1:
                    st.subheader("Clinical Summary")
                    
                    # More nuanced risk classification
                    if hasattr(model, "predict_proba"):
                        if proba[1] > 0.7:
                            risk_level = "High"
                            st.error(f"**{risk_level} Risk of Hypertension** ({proba[1]:.1%} probability)")
                            st.warning("Consider immediate clinical evaluation")
                        elif proba[1] > 0.3:
                            risk_level = "Moderate"
                            st.warning(f"**{risk_level} Risk of Hypertension** ({proba[1]:.1%} probability)")
                            st.info("Consider additional screening")
                        else:
                            risk_level = "Low"
                            st.success(f"**{risk_level} Risk of Hypertension** ({proba[1]:.1%} probability)")
                            st.info("Routine monitoring recommended")
                    else:
                        risk_level = "High" if prediction == 1 else "Low"
                        st.write(f"**{risk_level} Risk of Hypertension** (probability not available)")
                    
                    # Risk Factors Present
                    st.markdown("**Key Risk Factors Identified:**")
                    risk_factors = {
                        'Age > 50': age > 50,
                        'BMI ≥ 30': bmi >= 30,
                        'High Salt Intake': scid > 3,
                        'Chronic Stress': los == "Chronic Stress",
                        'Current Smoker': smoking == "Yes",
                        'Alcohol > 2 drinks/day': alcohol > 28
                    }
                    
                    for factor, present in risk_factors.items():
                        if present:
                            st.markdown(f"- 🔴 {factor}")
                
                with col2:
                    # Probability Visualization (only if probabilities available)
                    if hasattr(model, "predict_proba"):
                        fig1, ax1 = plt.subplots(figsize=(8, 4))
                        ax1.bar(['Low Risk', 'High Risk'], proba, 
                               color=['#2ecc71', '#e74c3c'], width=0.6)
                        ax1.set_ylim(0, 1)
                        ax1.set_ylabel('Probability', fontsize=10)
                        ax1.set_title('Hypertension Risk Probability', pad=15, fontsize=12)
                        for i, v in enumerate(proba):
                            ax1.text(i, v + 0.02, f"{v:.1%}", 
                                    ha='center', fontsize=11, weight='bold')
                        st.pyplot(fig1)
                    else:
                        st.info("Probability visualization not available for this model")
                
                st.divider()
                
                # Rest of your code (Clinical Recommendations, Feature Importance, etc.)
                # ... [keep the remaining sections unchanged] ...
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.write("Technical details:", str(e))
