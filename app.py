import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
import firebase_admin
from firebase_admin import credentials, firestore, auth
from google.cloud.firestore_v1.base_query import FieldFilter
import json
from streamlit.components.v1 import html
from firebase_admin import exceptions as firebase_exceptions

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

# Initialize Firebase (cached to prevent multiple initializations)
@st.cache_resource
def init_firebase():
    try:
        if not firebase_admin._apps:
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

# Firebase Client SDK Authentication
def firebase_auth_script():
    firebase_config = {
        "apiKey": st.secrets["firebase"]["api_key"],
        "authDomain": f"{st.secrets['firebase']['project_id']}.firebaseapp.com",
        "projectId": st.secrets["firebase"]["project_id"],
        "storageBucket": f"{st.secrets['firebase']['project_id']}.appspot.com",
        "messagingSenderId": st.secrets["firebase"]["messaging_sender_id"],
        "appId": st.secrets["firebase"]["app_id"]
    }
    
    return f"""
    <script src="https://www.gstatic.com/firebasejs/9.6.0/firebase-app-compat.js"></script>
    <script src="https://www.gstatic.com/firebasejs/9.6.0/firebase-auth-compat.js"></script>
    <script>
        const firebaseConfig = {json.dumps(firebase_config)};
        const app = firebase.initializeApp(firebaseConfig);
        const auth = app.auth();
        
        function signIn(email, password) {{
            return auth.signInWithEmailAndPassword(email, password);
        }}
        
        function signUp(email, password, name) {{
            return auth.createUserWithEmailAndPassword(email, password)
                .then((userCredential) => {{
                    return userCredential.user.updateProfile({{ displayName: name }});
                }});
        }}
        
        function signOut() {{
            return auth.signOut();
        }}
        
        function getCurrentUser() {{
            return new Promise((resolve) => {{
                const unsubscribe = auth.onAuthStateChanged(user => {{
                    unsubscribe();
                    resolve(user);
                }});
            }});
        }}
        
        // Make functions available to Streamlit
        window.firebaseAuth = {{ signIn, signUp, signOut, getCurrentUser }};
    </script>
    """

# Initialize Firebase Client SDK
html(firebase_auth_script())

# Authentication functions using Firebase Client SDK
def sign_in(email, password):
    try:
        js = f"""
        window.firebaseAuth.signIn('{email}', '{password}')
            .then(() => {{
                window.parent.postMessage({{type: 'AUTH_SUCCESS'}}, '*');
            }})
            .catch(error => {{
                window.parent.postMessage({{type: 'AUTH_ERROR', error: error.message}}, '*');
            }});
        """
        html(js)
        return True
    except Exception as e:
        st.error(f"Sign in error: {e}")
        return False

def sign_up(email, password, name):
    try:
        js = f"""
        window.firebaseAuth.signUp('{email}', '{password}', '{name}')
            .then(() => {{
                window.parent.postMessage({{type: 'SIGNUP_SUCCESS'}}, '*');
            }})
            .catch(error => {{
                window.parent.postMessage({{type: 'AUTH_ERROR', error: error.message}}, '*');
            }});
        """
        html(js)
        return True
    except Exception as e:
        st.error(f"Sign up error: {e}")
        return False

def sign_out():
    try:
        js = """
        window.firebaseAuth.signOut()
            .then(() => {
                window.parent.postMessage({type: 'SIGNOUT_SUCCESS'}, '*');
            });
        """
        html(js)
        return True
    except Exception as e:
        st.error(f"Sign out error: {e}")
        return False

def get_current_user():
    try:
        js = """
        window.firebaseAuth.getCurrentUser()
            .then(user => {
                if (user) {
                    window.parent.postMessage({
                        type: 'CURRENT_USER',
                        user: {
                            uid: user.uid,
                            email: user.email,
                            name: user.displayName
                        }
                    }, '*');
                } else {
                    window.parent.postMessage({type: 'NO_USER'}, '*');
                }
            });
        """
        html(js)
    except Exception as e:
        st.error(f"Error getting current user: {e}")

# Handle auth messages from JavaScript
def handle_auth_messages():
    auth_messages = st.session_state.get('auth_messages', [])
    for message in auth_messages:
        if message['type'] == 'AUTH_SUCCESS':
            st.session_state.user_updated = True
            st.rerun()
        elif message['type'] == 'SIGNUP_SUCCESS':
            st.success("Account created successfully! Please sign in.")
            st.session_state.show_signup = False
            st.session_state.show_login = True
        elif message['type'] == 'AUTH_ERROR':
            st.error(f"Authentication error: {message['error']}")
        elif message['type'] == 'CURRENT_USER':
            st.session_state.user = message['user']
        elif message['type'] == 'NO_USER':
            st.session_state.user = None
    
    if auth_messages:
        st.session_state.auth_messages = []

# Save prediction to Firestore with user association
def save_prediction_to_firestore(input_data, prediction, proba):
    try:
        db = init_firebase()
        if not db or 'user' not in st.session_state:
            return False
            
        predictions_ref = db.collection('hbp_predictions')
        
        prediction_data = {
            'user_id': st.session_state.user['uid'],
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
        
        doc_ref = predictions_ref.add(prediction_data)
        return True
        
    except Exception as e:
        st.error(f"Firestore save error: {e}")
        return False

# Get user's predictions from Firestore
def get_user_predictions(user_id):
    try:
        db = init_firebase()
        if not db:
            st.warning("Database connection failed")
            return None
            
        predictions_ref = db.collection('hbp_predictions')
        
        try:
            # Try the optimized query first
            docs = predictions_ref.where('user_id', '==', user_id)\
                                 .order_by('timestamp', direction=firestore.Query.DESCENDING)\
                                 .stream()
        except Exception as e:
            if "index" in str(e).lower():
                # If index isn't ready, use client-side sorting
                st.warning("Index not ready yet - using basic query (may be slower)")
                all_docs = predictions_ref.where('user_id', '==', user_id).stream()
                docs = sorted(
                    all_docs, 
                    key=lambda doc: doc.to_dict().get('timestamp'), 
                    reverse=True
                )
            else:
                raise e
        
        predictions = []
        for doc in docs:
            try:
                pred_data = doc.to_dict()
                pred_data['id'] = doc.id
                predictions.append(pred_data)
            except Exception as e:
                st.warning(f"Skipping corrupt document: {e}")
                continue
                
        return predictions if predictions else None
        
    except Exception as e:
        st.error(f"Error fetching predictions: {str(e)[:200]}")  # Truncate long errors
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

# Initialize session state
if 'user' not in st.session_state:
    st.session_state.user = None
if 'submitted' not in st.session_state:
    st.session_state.submitted = False
if 'show_login' not in st.session_state:
    st.session_state.show_login = False
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'auth_messages' not in st.session_state:
    st.session_state.auth_messages = []
if 'user_updated' not in st.session_state:
    st.session_state.user_updated = False

# JavaScript to handle auth state changes
auth_state_js = """
<script>
    // Listen for auth state changes
    window.firebaseAuth.getCurrentUser();
    
    // Listen for messages from the auth functions
    window.addEventListener('message', (event) => {
        if (event.data.type && event.data.type.startsWith('AUTH_')) {
            window.parent.postMessage(event.data, '*');
        }
    });
    
    // Forward messages to Streamlit
    window.addEventListener('message', (event) => {
        if (event.data.type && event.data.type.startsWith('AUTH_')) {
            const streamlitMsg = {
                isStreamlitMessage: true,
                type: 'AUTH_UPDATE',
                data: event.data
            };
            window.parent.postMessage(streamlitMsg, '*');
        }
    });
</script>
"""

html(auth_state_js)

# Handle Streamlit messages
def handle_streamlit_messages():
    try:
        messages = st.session_state.get('streamlit_messages', [])
        for message in messages:
            if message.get('type') == 'AUTH_UPDATE':
                st.session_state.auth_messages.append(message['data'])
        
        if messages:
            st.session_state.streamlit_messages = []
            handle_auth_messages()
            
    except Exception as e:
        st.error(f"Error handling messages: {e}")

# JavaScript to receive messages from iframe
receive_messages_js = """
<script>
    window.addEventListener('message', (event) => {
        if (event.data.isStreamlitMessage) {
            const msg = event.data;
            if (msg.type === 'AUTH_UPDATE') {
                window.parent.streamlitAPI.enqueueMessage(msg);
            }
        }
    });
</script>
"""

html(receive_messages_js)

# Check authentication state on app load
if not st.session_state.user_updated:
    get_current_user()
    st.session_state.user_updated = True

handle_streamlit_messages()
handle_auth_messages()

# Sidebar authentication
with st.sidebar:
    st.title("Account")
    
    if st.session_state.user:
        st.markdown(f"Welcome, **{st.session_state.user['name']}**!")
        st.markdown(f"Email: {st.session_state.user['email']}")
        
        if st.button("Sign Out"):
            sign_out()
            st.session_state.user = None
            st.rerun()
        
        st.title("Menu")
        page = st.radio("Navigation", ["Predict", "View History", "Ontology", "About"])
    else:
        if st.button("Sign In"):
            st.session_state.show_login = True
            st.session_state.show_signup = False
        
        if st.button("Create Account"):
            st.session_state.show_signup = True
            st.session_state.show_login = False

# Main App Content
if st.session_state.user:
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
        - Data stored securely in Firebase Firestore
        """)

    # View History page
    elif page == "View History":
        st.title("Prediction History")
        
        with st.spinner('Loading your prediction history...'):
            predictions = get_user_predictions(st.session_state.user['uid'])
        
        if predictions:
            st.write(f"Found {len(predictions)} historical predictions")
            
            history_df = pd.DataFrame([{
                'Date': pred['timestamp'].strftime('%Y-%m-%d %H:%M'),
                'Age': pred['age'],
                'Gender': pred['gender'],
                'Prediction': pred['prediction'],
                'Probability': f"{pred['probability']:.1%}",
                'Risk Factors': ', '.join([k.replace('_', ' ') for k, v in pred['risk_factors'].items() if v])
            } for pred in predictions])
            
            st.dataframe(history_df, use_container_width=True)
            
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
                        
                        if save_prediction_to_firestore(input_data, prediction, proba):
                            st.success("Prediction saved to your history")
                        else:
                            st.warning("Prediction completed but not saved to history")

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

else:
    st.title("High Blood Pressure Risk Prediction System")
    st.warning("Please sign in or create an account to use the prediction tool")
    
    if st.session_state.show_login:
        with st.expander("Sign In", expanded=True):
            with st.form("main_sign_in_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign In")
                
                if submitted:
                    if sign_in(email, password):
                        st.rerun()
    
    if st.session_state.show_signup:
        with st.expander("Create Account", expanded=True):
            with st.form("main_sign_up_form"):
                name = st.text_input("Full Name")
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Sign Up")
                
                if submitted:
                    if password != confirm_password:
                        st.error("Passwords don't match")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        if sign_up(email, password, name):
                            st.rerun()

# Footer
st.markdown("""
<div class="custom-footer">
    High Blood Pressure Risk Prediction System © 2025 | Clinical Decision Support Tool
</div>
""", unsafe_allow_html=True)
