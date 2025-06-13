import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
from uuid import uuid4

# Initialize Firebase (only once)
def initialize_firebase():
    if not firebase_admin._apps:
        try:
            # Use Streamlit secrets for Firebase configuration
            firebase_cred = {
                "type": st.secrets["firebase"]["type"],
                "project_id": st.secrets["firebase"]["project_id"],
                "private_key_id": st.secrets["firebase"]["private_key_id"],
                "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'),
                "client_email": st.secrets["firebase"]["client_email"],
                "client_id": st.secrets["firebase"]["client_id"],
                "auth_uri": st.secrets["firebase"]["auth_uri"],
                "token_uri": st.secrets["firebase"]["token_uri"],
                "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
                "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
            }
            
            cred = credentials.Certificate(firebase_cred)
            firebase_admin.initialize_app(cred, {
                'databaseURL': st.secrets["firebase"]["databaseURL"]
            })
        except Exception as e:
            st.error(f"Failed to initialize Firebase: {str(e)}")
            return False
    return True

# Set page config
st.set_page_config(page_title="HBP Risk Prediction System", layout="wide")

# Hide Streamlit default UI and style footer
st.markdown("""
    <style>
    /* Hide default Streamlit UI */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton, .st-emotion-cache-13ln4jf, button[kind="icon"] {
        display: none !important;
    }
    .custom-footer {
        text-align: center;
        font-size: 14px;
        margin-top: 50px;
        padding: 20px;
        color: #aaa;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("Menu")
    page = st.radio("Go to", ["Predict", "Ontology", "About", "View Predictions"])

# Load model and scaler (cached)
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('hypertension_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Initialize Firebase
if not initialize_firebase():
    st.stop()

# Function to save prediction to Firebase
def save_prediction_to_firebase(input_data, prediction, proba):
    try:
        prediction_data = {
            'id': str(uuid4()),
            'input_data': input_data,
            'prediction': int(prediction[0]),
            'probability': float(proba[1]),  # Probability of high risk
            'timestamp': datetime.now().isoformat(),
            'recommendations': {
                'lifestyle': [],
                'monitoring': [],
                'referrals': []
            }
        }

        # Add recommendations
        if input_data['bmi'] >= 30:
            prediction_data['recommendations']['lifestyle'].append("Weight reduction program")
        if input_data['scid'] > 3:
            prediction_data['recommendations']['lifestyle'].append("Sodium restriction (<2g/day)")
        if input_data['pa'] < 1500:
            prediction_data['recommendations']['lifestyle'].append("Increase physical activity")
        if input_data['alcohol'] > 14:
            prediction_data['recommendations']['lifestyle'].append("Reduce alcohol consumption")
        
        if prediction[0] == 1:
            prediction_data['recommendations']['monitoring'].append("Weekly BP checks")
            prediction_data['recommendations']['monitoring'].append("Renal function tests")
        else:
            prediction_data['recommendations']['monitoring'].append("Annual screening")
        
        if input_data['pregnancy'] == 1:
            prediction_data['recommendations']['monitoring'].append("High-risk obstetric follow-up")
        
        if input_data['ckd'] == 1:
            prediction_data['recommendations']['referrals'].append("Nephrology consult")
        if input_data['atd'] == 1:
            prediction_data['recommendations']['referrals'].append("Endocrinology evaluation")
        if input_data['los'] == 3:  # Chronic Stress
            prediction_data['recommendations']['referrals'].append("Behavioral health referral")

        # Push data to Firebase
        ref = db.reference('predictions')
        new_prediction_ref = ref.push(prediction_data)
        return True, new_prediction_ref.key
    except Exception as e:
        return False, str(e)

# Function to get all predictions from Firebase
def get_predictions_from_firebase():
    try:
        ref = db.reference('predictions')
        predictions = ref.get()
        return True, predictions if predictions else {}
    except Exception as e:
        return False, str(e)

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
        st.error("Ontology image not found. Please ensure 'ontology.PNG' is in the same directory.")
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

# View Predictions page
elif page == "View Predictions":
    st.title("Stored Predictions")
    st.write("Viewing all predictions stored in the database")
    
    success, predictions = get_predictions_from_firebase()
    
    if not success:
        st.error(f"Failed to retrieve predictions: {predictions}")
    else:
        if not predictions:
            st.info("No predictions found in the database")
        else:
            st.write(f"Found {len(predictions)} predictions")
            
            for pred_id, pred_data in predictions.items():
                with st.expander(f"Prediction {pred_id[:8]}... - {pred_data['timestamp']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Input Data")
                        st.json(pred_data['input_data'])
                        
                    with col2:
                        st.subheader("Results")
                        st.write(f"**Prediction:** {'High Risk' if pred_data['prediction'] == 1 else 'Low Risk'}")
                        st.write(f"**Probability:** {pred_data['probability']:.1%}")
                        
                        st.subheader("Recommendations")
                        st.write("**Lifestyle Modifications:**")
                        for rec in pred_data['recommendations']['lifestyle']:
                            st.write(f"- {rec}")
                            
                        st.write("**Clinical Monitoring:**")
                        for rec in pred_data['recommendations']['monitoring']:
                            st.write(f"- {rec}")
                            
                        st.write("**Specialist Referrals:**")
                        for rec in pred_data['recommendations']['referrals']:
                            st.write(f"- {rec}")

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
                        'LOH': loh,
                        'GPC': gpc,
                        'Age': age,
                        'BMI': bmi,
                        'Sex': 1 if gender == "Female" else 0,
                        'Pregnancy': 1 if pregnancy == "Yes" else 0,
                        'Smoking': 1 if smoking == "Yes" else 0,
                        'PA': pa,
                        'salt_CID': scid,
                        'alcohol_CPD': alcohol,
                        'LOS': ["Acute/normal stress", "Episodic acute stress", "Chronic Stress"].index(los) + 1,
                        'CKD': 1 if ckd == "Yes" else 0,
                        'ATD': 1 if atd == "Yes" else 0
                    }

                    #features = ['loh', 'gpc', 'age', 'bmi', 'gender', 'pregnancy', 'smoking',
                                #'pa', 'scid', 'alcohol', 'los', 'ckd', 'atd']
                    features = ['LOH', 'GPC', 'Age', 'BMI', 'Sex', 'Pregnancy', 'Smoking',
                                'PA', 'salt_CID', 'alcohol_CPD', 'LOS', 'CKD', 'ATD']
                    X = pd.DataFrame([[input_data[feature] for feature in features]], columns=features)

                    X_scaled = scaler.transform(X)
                    prediction = model.predict(X_scaled)
                    proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]

                    # Save to Firebase
                    success, result = save_prediction_to_firebase(input_data, prediction, proba)
                    if not success:
                        st.error(f"Failed to save prediction: {result}")

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
                        'High Salt Intake': scid > 2325,
                        'Chronic Stress': los == "Chronic Stress",
                        'Current Smoker': smoking == "Yes",
                        'Alcohol > 2 drinks/day': alcohol > 355
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
                    if scid > 3:
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

                if hasattr(model, "feature_importances_"):
                    st.subheader("Key Predictive Factors")
                    importance = pd.DataFrame({
                        'Factor': features,
                        'Impact': model.feature_importances_
                    }).sort_values('Impact', ascending=False)

                    fig2, ax2 = plt.subplots(figsize=(8, 5))
                    ax2.barh(importance['Factor'][:8],
                             importance['Impact'][:8],
                             color=plt.cm.Blues(np.linspace(0.3, 1, 8)))
                    ax2.set_xlabel('Relative Importance', fontsize=10)
                    ax2.set_title('Top Contributing Factors', pad=15, fontsize=12)
                    st.pyplot(fig2)

                    with st.expander("View Complete Factor Analysis"):
                        st.dataframe(importance.set_index('Factor').style.background_gradient(cmap='Blues'))

                st.divider()

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# Footer
st.markdown("""
<div class="custom-footer">
    High Blood Pressure Risk Prediction System © 2023 | Clinical Decision Support Tool
</div>
""", unsafe_allow_html=True)
