import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import firebase_admin
from firebase_admin import credentials, auth, firestore
from datetime import datetime

# Initialize Firebase (only once)
if not firebase_admin._apps:
    try:
        # Get Firebase config from Streamlit secrets
        firebase_config = {
            "type": st.secrets["firebase"]["type"],
            "project_id": st.secrets["firebase"]["project_id"],
            "private_key_id": st.secrets["firebase"]["private_key_id"],
            "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'),
            "client_email": st.secrets["firebase"]["client_email"],
            "client_id": st.secrets["firebase"]["client_id"],
            "auth_uri": st.secrets["firebase"]["auth_uri"],
            "token_uri": st.secrets["firebase"]["token_uri"],
            "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
            "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"],
            "universe_domain": st.secrets["firebase"]["universe_domain"]
        }
        
        # Initialize Firebase
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"Firebase initialization error: {str(e)}")
        st.stop()

# Initialize Firestore
db = firestore.client()

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

# Authentication state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None

# Authentication functions
def firebase_signup(email, password):
    try:
        user = auth.create_user(
            email=email,
            password=password
        )
        return user.uid
    except Exception as e:
        st.error(f"Signup error: {e}")
        return None

def firebase_login(email, password):
    try:
        # In a real app, you would use Firebase Auth client SDK for web/mobile
        # This is a simplified version for Streamlit
        user = auth.get_user_by_email(email)
        
        # In production, you would verify the password properly
        # This is just for demonstration
        st.session_state.authenticated = True
        st.session_state.user_email = email
        st.session_state.user_id = user.uid
        return True
    except Exception as e:
        st.error(f"Login error: {e}")
        return False

def firebase_logout():
    st.session_state.authenticated = False
    st.session_state.user_email = None
    st.session_state.user_id = None

# Data storage functions
def save_prediction_to_firestore(user_id, input_data, prediction, probability):
    try:
        prediction_data = {
            'user_id': user_id,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'age': input_data['age'],
            'bmi': input_data['bmi'],
            'loh': input_data['loh'],
            'gpc': input_data['gpc'],
            'pa': input_data['pa'],
            'scid': input_data['scid'],
            'alcohol': input_data['alcohol'],
            'los': input_data['los'],
            'ckd': input_data['ckd'],
            'atd': input_data['atd'],
            'gender': input_data['gender'],
            'pregnancy': input_data['pregnancy'],
            'smoking': input_data['smoking'],
            'prediction_result': int(prediction[0]),
            'prediction_probability': float(probability[1]),
            'risk_factors': {
                'age_gt_50': input_data['age'] > 50,
                'bmi_gt_30': input_data['bmi'] >= 30,
                'high_salt': input_data['scid'] > 2325,
                'chronic_stress': input_data['los'] == "Chronic Stress",
                'smoker': input_data['smoking'] == "Yes",
                'high_alcohol': input_data['alcohol'] > 355
            }
        }
        
        db.collection('hbp_predictions').add(prediction_data)
        return True
    except Exception as e:
        st.error(f"Error saving prediction: {e}")
        return False

def get_user_predictions_from_firestore(user_id):
    try:
        predictions_ref = db.collection('hbp_predictions').where('user_id', '==', user_id).order_by('timestamp', direction=firestore.Query.DESCENDING)
        predictions = predictions_ref.stream()
        
        results = []
        for pred in predictions:
            pred_data = pred.to_dict()
            pred_data['id'] = pred.id
            results.append(pred_data)
        
        return results
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        return []

# Authentication form
def show_auth_form():
    auth_tab1, auth_tab2 = st.tabs(["Login", "Sign Up"])
    
    with auth_tab1:
        with st.form("login_form"):
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_pass")
            login_submitted = st.form_submit_button("Login")
            
            if login_submitted:
                if firebase_login(login_email, login_password):
                    st.rerun()

    with auth_tab2:
        with st.form("signup_form"):
            signup_email = st.text_input("Email", key="signup_email")
            signup_password = st.text_input("Password", type="password", key="signup_pass")
            signup_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
            signup_submitted = st.form_submit_button("Create Account")
            
            if signup_submitted:
                if signup_password != signup_confirm:
                    st.error("Passwords don't match")
                elif len(signup_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    user_id = firebase_signup(signup_email, signup_password)
                    if user_id:
                        st.success("Account created successfully! Please login.")
                        st.session_state.authenticated = True
                        st.session_state.user_email = signup_email
                        st.session_state.user_id = user_id
                        st.rerun()

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

# Sidebar navigation
with st.sidebar:
    if st.session_state.authenticated:
        st.title(f"Welcome, {st.session_state.user_email}")
        if st.button("Logout"):
            firebase_logout()
            st.rerun()
        
        st.divider()
        page = st.radio("Menu", ["Predict", "History", "Ontology", "About"])
    else:
        st.title("Menu")
        page = st.radio("Menu", ["Predict", "Ontology", "About"])

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

# Prediction History page
elif page == "History":
    if not st.session_state.authenticated:
        st.warning("Please login to view your prediction history")
    else:
        st.title("Your Prediction History")
        predictions = get_user_predictions_from_firestore(st.session_state.user_id)
        
        if not predictions:
            st.info("No predictions found in your history")
        else:
            st.write(f"Found {len(predictions)} predictions in your history")
            
            # Convert to DataFrame for better display
            history_data = []
            for pred in predictions:
                history_data.append({
                    "Date": pred['timestamp'].strftime("%Y-%m-%d %H:%M") if 'timestamp' in pred else "N/A",
                    "Age": pred['age'],
                    "Gender": pred['gender'],
                    "High Risk": "Yes" if pred['prediction_result'] == 1 else "No",
                    "Probability": f"{pred['prediction_probability']:.1%}",
                    "Risk Factors": ", ".join([k for k, v in pred['risk_factors'].items() if v])
                })
            
            history_df = pd.DataFrame(history_data)
            st.dataframe(history_df, hide_index=True, use_container_width=True)
            
            # Show some statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions", len(predictions))
            with col2:
                high_risk_count = sum(1 for p in predictions if p['prediction_result'] == 1)
                st.metric("High Risk Predictions", high_risk_count)
            with col3:
                avg_age = sum(p['age'] for p in predictions) / len(predictions)
                st.metric("Average Age", f"{avg_age:.1f} years")

# Prediction page
else:
    if not st.session_state.authenticated:
        st.title("Please Login to Use the Prediction Tool")
        show_auth_form()
        st.stop()
    
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
                        'gender': "Female" if gender == "Female" else "Male",
                        'pregnancy': pregnancy,
                        'smoking': smoking,
                        'pa': pa,
                        'scid': scid,
                        'alcohol': alcohol,
                        'los': los,
                        'ckd': ckd,
                        'atd': atd
                    }

                    features = ['loh', 'gpc', 'age', 'bmi', 'gender', 'pregnancy', 'smoking',
                                'pa', 'scid', 'alcohol', 'los', 'ckd', 'atd']
                    
                    # Prepare data for model
                    model_input = {
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
                    
                    X = pd.DataFrame([[model_input[feature] for feature in features]], columns=features)
                    X_scaled = scaler.transform(X)
                    prediction = model.predict(X_scaled)
                    proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]

                    # Save prediction to Firestore
                    save_prediction_to_firestore(st.session_state.user_id, input_data, prediction, proba)

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

# Custom footer
st.markdown("""
<div class="custom-footer">
    High Blood Pressure Risk Prediction System © 2023 | Clinical Decision Support Tool
</div>
""", unsafe_allow_html=True)
