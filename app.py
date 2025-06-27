import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
import hashlib
import binascii
import os

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

# Password hashing functions
def hash_password(password):
    """Hash a password for storing."""
    salt = hashlib.sha256(os.urandom(60)).hexdigest().encode('ascii')
    pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), 
                                salt, 100000)
    pwdhash = binascii.hexlify(pwdhash)
    return (salt + pwdhash).decode('ascii')

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    salt = stored_password[:64]
    stored_password = stored_password[64:]
    pwdhash = hashlib.pbkdf2_hmac('sha512', 
                                provided_password.encode('utf-8'), 
                                salt.encode('ascii'), 
                                100000)
    pwdhash = binascii.hexlify(pwdhash).decode('ascii')
    return pwdhash == stored_password

# Initialize Firebase (cached to prevent multiple initializations)
@st.cache_resource
def init_firebase():
    try:
        # Check if Firebase app is already initialized
        if not firebase_admin._apps:
            # Load credentials from secrets
            firebase_creds = {
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
            
            cred = credentials.Certificate(firebase_creds)
            firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.error(f"Firebase initialization error: {e}")
        return None

# Authentication functions
def login_user(email, password):
    try:
        db = init_firebase()
        
        # Query for user by email
        users_ref = db.collection('users')
        query = users_ref.where('email', '==', email).limit(1).get()
        
        if len(query) == 0:
            st.error("Invalid credentials")  # Generic error message
            return False
            
        user_doc = query[0]
        user_data = user_doc.to_dict()
        
        # Verify password
        if verify_password(user_data['password'], password):
            st.session_state.user = {
                'uid': user_doc.id,
                'email': email,
                'name': user_data.get('name')
            }
            return True
        else:
            st.error("Invalid credentials")
            return False
            
    except Exception as e:
        st.error(f"Login error: {e}")
        return False

def signup_user(email, password, name):
    try:
        db = init_firebase()
        
        # Check if user already exists
        users_ref = db.collection('users')
        query = users_ref.where('email', '==', email).limit(1).get()
        
        if len(query) > 0:
            st.error("Email already exists")
            return False
        
        # Hash the password before storing
        hashed_password = hash_password(password)
        
        # Create user document
        user_data = {
            'email': email,
            'name': name,
            'password': hashed_password,
            'created_at': datetime.datetime.now(datetime.timezone.utc)
        }
        
        # Add a new document with auto-generated ID
        doc_ref = users_ref.add(user_data)
        
        # Set session state
        st.session_state.user = {
            'uid': doc_ref[1].id,  # The auto-generated document ID
            'email': email,
            'name': name
        }
        return True
        
    except Exception as e:
        st.error(f"Signup error: {e}")
        return False

# Save prediction to Firestore with user ID
def save_prediction_to_firestore(input_data, prediction, proba):
    try:
        db = init_firebase()
        if not db:
            return False
            
        predictions_ref = db.collection('hbp_predictions')
        
        prediction_data = {
            'age': input_data['Age'],
            'bmi': input_data['BMI'],
            'loh': input_data['LOH'],
            'gpc': input_data['GPC'],
            'physical_activity': input_data['PA'],
            'salt_intake': input_data['salt_CID'],
            'alcohol': input_data['alcohol_CPD'],
            'stress_level': input_data['LOS'],
            'ckd': input_data['CKD'],
            'atd': input_data['ATD'],
            'gender': 'Female' if input_data['Sex'] == 1 else 'Male',
            'pregnancy': 'Yes' if input_data['Pregnancy'] == 1 else 'No',
            'smoking': 'Yes' if input_data['Smoking'] == 1 else 'No',
            'prediction': 'High Risk' if prediction[0] == 1 else 'Low Risk',
            'probability': float(proba[1]),
            'timestamp': datetime.datetime.now(datetime.timezone.utc),
            'risk_factors': {
                'age_gt_50': input_data['Age'] > 50,
                'bmi_gt_30': input_data['BMI'] >= 30,
                'high_salt': input_data['salt_CID'] > 2325,
                'chronic_stress': input_data['LOS'] == 3,
                'smoker': input_data['Smoking'] == 1,
                'heavy_drinker': input_data['alcohol_CPD'] > 355
            }
        }
        
        # Add user ID if logged in
        if 'user' in st.session_state:
            prediction_data['user_id'] = st.session_state.user['uid']
        
        # Add a new document with auto-generated ID
        doc_ref = predictions_ref.add(prediction_data)
        return True
        
    except Exception as e:
        st.error(f"Firestore save error: {e}")
        return False

# Get predictions from Firestore (filter by user if logged in)
def get_predictions():
    try:
        db = init_firebase()
        if not db:
            return None
            
        predictions_ref = db.collection('hbp_predictions')
        
        # If user is logged in, only get their predictions
        if 'user' in st.session_state:
            query = predictions_ref.where('user_id', '==', st.session_state.user['uid'])
        else:
            query = predictions_ref
            
        docs = query.order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
        
        predictions = []
        for doc in docs:
            pred_data = doc.to_dict()
            pred_data['id'] = doc.id
            predictions.append(pred_data)
            
        return predictions
    except Exception as e:
        st.error(f"Error fetching predictions: {e}")
        return None

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

# Initialize session state for authentication
if 'user' not in st.session_state:
    st.session_state.user = None
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

# Sidebar - Authentication and Navigation
with st.sidebar:
    if st.session_state.user:
        st.success(f"Logged in as {st.session_state.user['email']}")
        if st.button("Logout"):
            st.session_state.user = None
            st.session_state.submitted = False
            st.rerun()
    else:
        st.title("Authentication")
        auth_tab = st.tabs(["Login", "Sign Up"])
        
        with auth_tab[0]:  # Login tab
            with st.form("login_form"):
                login_email = st.text_input("Email", key="login_email")
                login_password = st.text_input("Password", type="password", key="login_password")
                login_submitted = st.form_submit_button("Login")
                
                if login_submitted:
                    if login_user(login_email, login_password):
                        st.success("Login successful!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
        
        with auth_tab[1]:  # Signup tab
            with st.form("signup_form"):
                signup_name = st.text_input("Full Name", key="signup_name")
                signup_email = st.text_input("Email", key="signup_email")
                signup_password = st.text_input("Password", type="password", key="signup_password")
                confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
                signup_submitted = st.form_submit_button("Sign Up")
                
                if signup_submitted:
                    if signup_password != confirm_password:
                        st.error("Passwords don't match")
                    elif len(signup_password) < 8:
                        st.error("Password must be at least 8 characters")
                    elif signup_user(signup_email, signup_password, signup_name):
                        st.success("Account created successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Error creating account")

    # Navigation menu (only show if logged in)
    if st.session_state.user:
        st.title("Menu")
        page = st.radio("Go to", ["Predict", "View History", "Ontology", "About"])
    else:
        st.title("")
        # st.write("""
        # ### High Blood Pressure Risk Prediction System
        # Please login or sign up to access the prediction tool.
        
        # **Features**:
        # - Personalized risk assessment
        # - Historical prediction tracking
        # - Detailed ontology visualization
        # - Clinical recommendations
        # """)

# Main content area
if not st.session_state.user:
    st.title("Welcome to HBP Risk Prediction System")
    st.write("""
    ## About This System
    
    This clinical decision support tool helps healthcare professionals assess the risk of 
    hypertension (high blood pressure) in patients based on multiple risk factors.
    
    **Key Features**:
    - Machine learning model with 85% accuracy
    - 13 key risk factors analyzed
    - Personalized care recommendations
    - Secure data storage
    
    Please login or sign up using the sidebar to access the prediction tools.
    """)
    
    st.image("https://www.heart.org/-/media/Images/Health-Topics/High-Blood-Pressure/BPLevels.png", 
             caption="Blood Pressure Classification", use_column_width=True)
    
    st.write("""
    ### Why Early Detection Matters
    Hypertension is a major risk factor for heart disease, stroke, and kidney failure. 
    Early detection and intervention can significantly reduce these risks.
    """)

elif page == "Ontology":
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
    - Data stored securely in Firebase Firestore
    """)

elif page == "View History":
    st.title("Prediction History")
    
    with st.spinner('Loading prediction history...'):
        predictions = get_predictions()
    
    if predictions:
        st.write(f"Found {len(predictions)} historical predictions")
        
        # Convert to DataFrame for display
        history_df = pd.DataFrame([{
            'Date': pred['timestamp'].strftime('%Y-%m-%d %H:%M'),
            'Age': pred['age'],
            'Gender': pred['gender'],
            'Prediction': pred['prediction'],
            'Probability': f"{pred['probability']:.1%}",
            'Risk Factors': ', '.join([k.replace('_', ' ') for k, v in pred['risk_factors'].items() if v])
        } for pred in predictions])
        
        st.dataframe(history_df, use_container_width=True)
        
        # Show detailed view
        selected_pred = st.selectbox("Select a prediction to view details:", 
                                   range(len(predictions)), 
                                   format_func=lambda x: f"Prediction {x+1} - {predictions[x]['timestamp'].strftime('%Y-%m-%d %H:%M')}")
        
        if selected_pred is not None:
            pred = predictions[selected_pred]
            with st.expander("Detailed View", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Patient Details")
                    st.write(f"**Age**: {pred['age']}")
                    st.write(f"**Gender**: {pred['gender']}")
                    st.write(f"**BMI**: {pred['bmi']:.1f}")
                    st.write(f"**Pregnancy**: {pred['pregnancy']}")
                    
                with col2:
                    st.subheader("Lifestyle Factors")
                    st.write(f"**Smoking**: {pred['smoking']}")
                    st.write(f"**Alcohol (ml/day)**: {pred['alcohol']}")
                    st.write(f"**Physical Activity**: {pred['physical_activity']}")
                    st.write(f"**Salt Intake (g)**: {pred['salt_intake']:.1f}")
                
                st.subheader("Clinical Factors")
                col3, col4 = st.columns(2)
                with col3:
                    st.write(f"**Chronic Kidney Disease**: {'Yes' if pred['ckd'] else 'No'}")
                    st.write(f"**Thyroid Disorders**: {'Yes' if pred['atd'] else 'No'}")
                with col4:
                    st.write(f"**Hemoglobin Level**: {pred['loh']:.1f} g/dl")
                    st.write(f"**Inbreeding Coefficient**: {pred['gpc']:.2f}")
                
                st.subheader("Prediction Results")
                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(['Low Risk', 'High Risk'], 
                      [1 - pred['probability'], pred['probability']], 
                      color=['#2ecc71', '#e74c3c'])
                ax.set_ylim(0, 1)
                ax.set_title('Risk Probability')
                st.pyplot(fig)
    else:
        st.warning("No prediction history found")

# Prediction page
else:
    st.title("High Blood Pressure Risk Prediction Tool")
    st.write("""
    This tool predicts the risk of HBP based on patient characteristics.
    Please fill in all the fields below and click 'Predict'.
    """)

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

                    features = ['LOH', 'GPC', 'Age', 'BMI', 'Sex', 'Pregnancy', 'Smoking',
                                'PA', 'salt_CID', 'alcohol_CPD', 'LOS', 'CKD', 'ATD']
                    X = pd.DataFrame([[input_data[feature] for feature in features]], columns=features)

                    X_scaled = scaler.transform(X)
                    prediction = model.predict(X_scaled)
                    proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
                    
                    # Save to Firestore
                    if save_prediction_to_firestore(input_data, prediction, proba):
                        st.success("Prediction saved to database")
                    else:
                        st.warning("Prediction completed but not saved to database")

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
    High Blood Pressure Risk Prediction System © 2025 | Clinical Decision Support Tool
</div>
""", unsafe_allow_html=True)
