import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

# Set page config
st.set_page_config(page_title="HBP Risk Prediction Syetem", layout="wide")

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
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Ontology page
if page == "Ontology":
    st.title("Ontology For HBP Prediction System")
    st.write("""
    ### Key Concepts and Relationships

    **Risk Factor Categories**:
    - **(BPA)**: Blood Pressure Abnormalities.
    - **Demographic**: Age, Gender, Pregnancy Status
    - **Lifestyle**: Smoking, Alcohol Consumption, Physical Activity
    - **Clinical**: Chronic Kidney Disease, Thyroid Disorders
    - **Biochemical**: Hemoglobin Levels, Salt Intake
    - **Genetic**: Inbreeding Coefficient
    """)
    
    # Display the local ontology image
    try:
        ontology_image = "ontology.PNG"  # Make sure this matches your filename exactly
        st.image(ontology_image,
                caption="BPA Risk Factor Ontology Diagram",
                use_column_width=True)  # Adjusts to column width
    except FileNotFoundError:
        st.error("Ontology image not found. Please ensure 'ontology.PNG' is in the same directory.")
    except Exception as e:
        st.error(f"Error loading ontology image: {e}")
    

# About page
elif page == "About":
    st.title("About This Tool")
    st.write("""
    ### High Blood Pressure Risk Prediction Tool
    
    **Version**: 1.0.0  
    **Purpose**: Clinical decision support for Blood Pressure Abnormalities exposure risk assessment
    
    **Methodology**:
    - Machine learning model trained on 2,000+ patient records
    - Validated with 85% accuracy
    - Incorporates 13 key risk factors
     
    """)

# Main Prediction Page
else:
    st.title("Hight Blood Pressure Risk Prediction Tool")
    st.write("""
    This tool predicts the risk of BPA exposure based on patient characteristics.
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

   # ... (previous imports and setup remain the same until the prediction logic)

    # Prediction logic
    if submitted:
        # Validate all fields are selected
        if (los == "Select" or ckd == "Select" or atd == "Select" or 
            gender == "Select Gender" or pregnancy == "Select" or smoking == "Select"):
            st.error("Please fill in all fields before submitting")
        else:
            try:
                # Show spinner while processing
                with st.spinner('Analyzing health data and calculating risk...'):
                    time.sleep(1)  # Simulate processing time
                    
                    # Encode inputs (same as before)
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
                    
                    prediction = model.predict(input_values)
                    
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(input_values)[0]
                    else:
                        proba = [0.5, 0.5]
                
                # Display results
                st.subheader("Prediction Result")
                if prediction[0] == 1:
                    st.error("**High risk of BPA**")
                    st.warning("This patient shows characteristics associated with higher BPA risk. Consider additional screening.")
                else:
                    st.success("**Low risk of Blood Pressure Abnormalities**")
                    st.info("This patient shows characteristics associated with lower HBP risk.")
                
                st.markdown("---")
                
                # Create columns for centered layout
                col1, col2 = st.columns([1, 2])  # Wider right column for charts
                
                with col2:
                    # Probability chart - middle sized
                    st.subheader("Risk Probability Distribution")
                    fig1, ax1 = plt.subplots(figsize=(8, 4))  # Adjusted figure size
                    ax1.bar(['Low Risk', 'High Risk'], proba, color=['green', 'red'])
                    ax1.set_ylim(0, 1)
                    ax1.set_ylabel('Probability')
                    ax1.set_title('HBP Risk Probability', pad=20)
                    for i, v in enumerate(proba):
                        ax1.text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=10)
                    st.pyplot(fig1)
                    
                    # Feature importance - middle sized
                    if hasattr(model, "feature_importances_"):
                        st.subheader("Feature Importance Analysis")
                        importance = model.feature_importances_
                        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance})
                        feature_importance = feature_importance.sort_values('Importance', ascending=False)
                        
                        fig2, ax2 = plt.subplots(figsize=(8, 5))  # Adjusted figure size
                        ax2.barh(feature_importance['Feature'], 
                                feature_importance['Importance'], 
                                color='skyblue')
                        ax2.set_xlabel('Importance Score', fontsize=10)
                        ax2.set_title('Relative Feature Importance', pad=20, fontsize=12)
                        ax2.tick_params(axis='both', which='major', labelsize=9)
                        st.pyplot(fig2)
                        
                        st.write("Top contributing factors:")
                        st.dataframe(feature_importance.set_index('Feature').style.format({'Importance': '{:.2f}'}))
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
