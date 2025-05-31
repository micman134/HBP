import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import shap

# Set page config
st.set_page_config(page_title="HBP Risk Prediction System", layout="wide")

# Sidebar navigation
with st.sidebar:
    st.title("Menu")
    page = st.radio("Go to", ["Predict", "Ontology", "About"])

# Load model and scaler (cached)
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('hypertension_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
    # Create SHAP explainer (TreeExplainer works for tree models like RandomForest, XGBoost)
    explainer = shap.TreeExplainer(model)
except Exception as e:
    st.error(f"Error loading model, scaler or creating explainer: {e}")
    st.stop()

# Ontology page
if page == "Ontology":
    st.title("Ontology For HBP Prediction System")
    st.write("""
    ### Key Concepts and Relationships
    **Risk Factor Categories**:
    - **Demographic**: Age, Gender, Pregnancy Status
    - **Lifestyle**: Smoking, Alcohol Consumption, Physical Activity
    - **Clinical**: Chronic Kidney Disease, Thyroid Disorders
    - **Biochemical**: Hemoglobin Levels, Salt Intake
    - **Genetic**: Inbreeding Coefficient
    """)
    try:
        st.image("ontology2.png", caption="HBP Risk Factor Ontology Diagram", use_column_width=True)
    except FileNotFoundError:
        st.error("Ontology image not found. Please ensure 'ontology2.png' is in the same directory.")
    except Exception as e:
        st.error(f"Error loading ontology image: {e}")

    try:
        st.image("ontology.PNG", caption="Protege Ontology Diagram", use_column_width=True)
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
    **Purpose**: Clinical decision support for Blood Pressure Abnormalities risk assessment  
    **Methodology**:
    - Machine learning model trained on 2,000+ patient records
    - Validated with 85% accuracy
    - Incorporates 13 key risk factors
    """)

# Prediction page
else:
    st.title("High Blood Pressure Risk Prediction Tool")
    st.write("""
    This tool predicts the risk of HBP based on patient characteristics.
    Please fill in all the fields below and click 'Predict'.
    """)

    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age (years)", 0, 120, 45)
            bmi = st.number_input("Body Mass Index (kg/m²)", 10.0, 50.0, 25.0, step=0.1)
            loh = st.number_input("Level Of Haemoglobin (Hb g/dl)", 0.0, 20.0, 12.0, step=0.1)
            gpc = st.number_input("Inbreed Coefficient", 0.0, 1.0, 0.0, step=0.01)
            pa = st.number_input("Physical Activity (CAL/4.18Kj)", 0, value=2000)
            scid = st.number_input("Salt Content in diet (grams)", 0.0, value=5.0, step=0.1)

        with col2:
            alcohol = st.number_input("Alcohol Consumption Per Day (ml)", 0, value=0)
            los = st.selectbox("Level Of Stress", 
                               ["Select", "Acute/normal stress", "Episodic acute stress", "Chronic Stress"], index=0)
            ckd = st.selectbox("Chronic Kidney Disease", ["Select", "Yes", "No"], index=0)
            atd = st.selectbox("Adrenal and Thyroid Disorders", ["Select", "Yes", "No"], index=0)
            gender = st.selectbox("Gender", ["Select Gender", "Female", "Male"], index=0)
            pregnancy = st.selectbox("Pregnancy Status", ["Select", "Yes", "No"], index=0)
            smoking = st.selectbox("Smoking Status", ["Select", "Yes", "No"], index=0)

        submitted = st.form_submit_button("Predict HBP Risk")
        if submitted:
            st.session_state.submitted = True

    if st.session_state.get('submitted', False):
        if "Select" in [los, ckd, atd, gender, pregnancy, smoking] or gender == "Select Gender":
            st.error("Please fill in all fields before submitting")
            st.session_state.submitted = False
        else:
            try:
                with st.spinner('Analyzing health data and calculating risk...'):
                    time.sleep(1)

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
                    X = pd.DataFrame([[input_data[feature] for feature in features]], columns=features)

                    X_scaled = scaler.transform(X)
                    prediction = model.predict(X_scaled)
                    proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]

                st.divider()
                col1, col2 = st.columns([1, 1.5])

                with col1:
                    st.subheader("Clinical Summary")
                    if prediction[0] == 1:
                        st.error(f"**High Risk of Hypertension**")
                        st.warning("Consider immediate clinical evaluation")
                    else:
                        st.success(f"**Low Risk of Hypertension**")
                        st.info("Routine monitoring recommended")

                    st.markdown("**Key Risk Factors Identified:**")
                    risk_factors = {
                        'Age > 50': age > 50,
                        'BMI ≥ 30': bmi >= 30,
                        'High Salt Intake': scid > 2.325,  # Corrected threshold to grams (was 2325 grams before)
                        'Chronic Stress': los == "Chronic Stress",
                        'Current Smoker': smoking == "Yes",
                        'Alcohol > 2 drinks/day': alcohol > 28  # 2 drinks ~ 28 ml (approx)
                    }

                    for factor, present in risk_factors.items():
                        if present:
                            st.markdown(f"- 🔴 {factor}")

                with col2:
                    fig1, ax1 = plt.subplots(figsize=(8, 4))
                    ax1.bar(['Low Risk', 'High Risk'], proba, color=['#2ecc71', '#e74c3c'], width=0.6)
                    ax1.set_ylim(0, 1)
                    ax1.set_ylabel('Probability', fontsize=10)
                    ax1.set_title('Hypertension Risk Probability', pad=15, fontsize=12)
                    ax1.set_yticks([0, 1])

                    for i, v in enumerate(proba):
                        ax1.text(i, v + 0.02, f"{v:.1%}", ha='center', fontsize=11, weight='bold')

                    st.pyplot(fig1)

                st.divider()
                st.subheader("Personalized Care Plan")

                rec_cols = st.columns(3)
                with rec_cols[0]:
                    st.markdown("**Lifestyle Modifications**")
                    if bmi >= 30:
                        st.write("- Weight reduction program")
                    if scid > 2:
                        st.write("- Sodium restriction (<2g/day)")
                    if pa < 1500:
                        st.write("- Increase physical activity")
                    if alcohol > 14:
                        st.write("- Reduce alcohol consumption")

                with rec_cols[1]:
                    st.markdown("**Clinical Monitoring**")
                    if prediction[0] == 1:
                        st.write("- Weekly BP checks")
                        st.write("- Renal function tests")
                    else:
                        st.write("- Annual screening")
                    if pregnancy == "Yes":
                        st.write("- High-risk obstetric follow-up")

                with rec_cols[2]:
                    st.markdown("**Specialist Referrals**")
                    if ckd == "Yes":
                        st.write("- Nephrology consult")
                    if atd == "Yes":
                        st.write("- Endocrinology evaluation")
                    if los == "Chronic Stress":
                        st.write("- Behavioral health referral")

                st.divider()

                # Personalized SHAP-based explanation
                if hasattr(model, "feature_importances_"):
                    st.subheader("Personalized Key Predictive Factors")

                    # Compute SHAP values for current input
                    shap_values = explainer.shap_values(X_scaled)
                    instance_shap_values = shap_values[1][0]

                    shap_df = pd.DataFrame({
                        'Feature': features,
                        'SHAP Value': instance_shap_values
                    })
                    shap_df['abs_shap'] = shap_df['SHAP Value'].abs()
                    shap_df = shap_df.sort_values(by='abs_shap', ascending=True)

                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in shap_df['SHAP Value']]
                    ax2.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors)
                    ax2.set_xlabel('SHAP Value (Impact on Prediction)')
                    ax2.set_title('Personalized Key Predictive Factors')
                    st.pyplot(fig2)

                    st.markdown("""
                    - **Green bars** increase risk prediction.
                    - **Red bars** decrease risk prediction.
                    """)

            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.session_state.submitted = False
