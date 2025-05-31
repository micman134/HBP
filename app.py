import streamlit as st
import joblib
import pandas as pd

# Set page config
st.set_page_config(page_title="BPA Risk Prediction", layout="wide")

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    return joblib.load('model.joblib')

# Load the model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Create the app
st.title("BPA Risk Prediction Tool")
st.write("""
This tool predicts the risk of BPA based on patient characteristics.
Please fill in all the fields below and click 'Predict'.
""")

# Input form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=45)
        bmi = st.number_input("Body Mass Index (kg/m²)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        loh = st.number_input("Level Of Haemoglobin (Hb g/dl)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        gpc = st.number_input("Inbreed Coefficient", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        pa = st.number_input("Physical Activity (CAL/4.18Kj)", min_value=0, value=2000)
        scid = st.number_input("Salt Content in diet (grams)", min_value=0.0, value=5.0, step=0.1)
        
    with col2:
        alcohol = st.number_input("Alcohol Consumption Per Day (millilitres)", min_value=0, value=0)
        los = st.selectbox("Level Of Stress", 
                          ["Select", "Acute/normal stress", "Episodic acute stress", "Chronic Stress"],
                          index=0)
        ckd = st.selectbox("Chronic Kidney Disease", ["Select", "Yes", "No"], index=0)
        atd = st.selectbox("Adrenal and Thyroid Disorders", ["Select", "Yes", "No"], index=0)
        gender = st.selectbox("Gender", ["Select Gender", "Female", "Male"], index=0)
        pregnancy = st.selectbox("Pregnancy Status", ["Select", "Yes", "No"], index=0)
        smoking = st.selectbox("Smoking Status", ["Select", "Yes", "No"], index=0)
    
    submitted = st.form_submit_button("Predict BPA Risk")

# Prediction logic
if submitted:
    # Validate all fields are selected
    if (los == "Select" or ckd == "Select" or atd == "Select" or 
        gender == "Select Gender" or pregnancy == "Select" or smoking == "Select"):
        st.error("Please fill in all fields before submitting")
    else:
        try:
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
            
            # Convert to correct order for model
            features = ['loh', 'gpc', 'age', 'bmi', 'gender', 'pregnancy', 'smoking', 
                       'pa', 'scid', 'alcohol', 'los', 'ckd', 'atd']
            input_values = [[input_data[feature] for feature in features]]
            
            # Make prediction
            prediction = model.predict(input_values)
            
            # Display results
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.error("**High risk of BPA**")
                st.warning("This patient shows characteristics associated with higher BPA risk. Consider additional screening.")
            else:
                st.success("**Low risk of BPA**")
                st.info("This patient shows characteristics associated with lower BPA risk.")
            
            # Add some space
            st.markdown("---")
            
            # Show probability if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_values)[0]
                st.subheader("Risk Probability")
                st.write(f"Probability of low risk: {proba[0]:.1%}")
                st.write(f"Probability of high risk: {proba[1]:.1%}")
                st.progress(proba[1])
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
