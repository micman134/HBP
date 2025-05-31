import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
                
                # Probability chart
                st.subheader("Risk Probability Distribution")
                fig1, ax1 = plt.subplots()
                ax1.bar(['Low Risk', 'High Risk'], proba, color=['green', 'red'])
                ax1.set_ylim(0, 1)
                ax1.set_ylabel('Probability')
                ax1.set_title('BPA Risk Probability')
                for i, v in enumerate(proba):
                    ax1.text(i, v + 0.02, f"{v:.1%}", ha='center')
                st.pyplot(fig1)
                
                st.write(f"Probability of low risk: {proba[0]:.1%}")
                st.write(f"Probability of high risk: {proba[1]:.1%}")
                st.progress(proba[1])
                
            # Feature importance if available
            if hasattr(model, "feature_importances_"):
                st.subheader("Feature Importance Analysis")
                importance = model.feature_importances_
                feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
                feature_importance = feature_importance.sort_values('Importance', ascending=False)
                
                fig2, ax2 = plt.subplots()
                ax2.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
                ax2.set_xlabel('Importance Score')
                ax2.set_title('Relative Feature Importance')
                st.pyplot(fig2)
                
                st.write("Top contributing factors to this prediction:")
                st.dataframe(feature_importance.set_index('Feature'))
            
            # Risk factor analysis
            st.subheader("Risk Factor Analysis")
            risk_factors = []
            protective_factors = []
            
            # Define thresholds for what constitutes risk/protective factors
            thresholds = {
                'age': (50, "Age > 50 years"),
                'bmi': (30, "BMI > 30 (Obese)"),
                'loh': (10, "Low Hemoglobin (<10 g/dl)"),
                'gpc': (0.25, "High Inbreeding Coefficient (>0.25)"),
                'pa': (1000, "Low Physical Activity (<1000 CAL)"),
                'scid': (3, "High Salt Intake (>3g)"),
                'alcohol': (20, "High Alcohol Consumption (>20ml)"),
                'los': (2, "Chronic Stress"),
                'ckd': (0.5, "Chronic Kidney Disease"),
                'atd': (0.5, "Adrenal/Thyroid Disorders"),
                'smoking': (0.5, "Smoking"),
                'pregnancy': (0.5, "Pregnancy")
            }
            
            for feature, value in input_data.items():
                if feature in thresholds:
                    threshold, message = thresholds[feature]
                    if isinstance(threshold, (int, float)):
                        if value > threshold:
                            risk_factors.append(message)
                        elif value < threshold/2:  # Arbitrary protective threshold
                            protective_factors.append(f"Low {message.split(' ')[0]}")
                    elif value == 1:  # For binary features
                        risk_factors.append(message)
            
            if risk_factors:
                st.warning("**Identified Risk Factors:**")
                for factor in risk_factors:
                    st.write(f"- {factor}")
            else:
                st.info("No significant risk factors identified")
                
            if protective_factors:
                st.success("**Identified Protective Factors:**")
                for factor in protective_factors:
                    st.write(f"- {factor}")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
