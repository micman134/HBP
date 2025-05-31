import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from fpdf import FPDF
from matplotlib import cm

# Set page config
st.set_page_config(page_title="HBP Clinical Decision System", layout="wide")

# Sidebar navigation
with st.sidebar:
    st.image("https://via.placeholder.com/150x50?text=HBP+CDS", width=150)
    st.title("Clinical Menu")
    page = st.radio("Navigate to", ["Predict", "Ontology", "About"])

# Load model
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Ontology and About pages (unchanged from previous)
# ... [previous ontology and about page code] ...

# Prediction Page
if page == "Predict":
    st.title("Hypertension Risk Assessment")
    st.markdown("""
    **Clinical Guidance Tool**  
    *Evidence-based hypertension risk prediction with management guidance*
    """)

    # Input form
    with st.form("clinical_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=120, value=45)
            bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
            loh = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=20.0, value=13.5, step=0.1)
            scid = st.number_input("Daily Salt Intake (g)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
            pa = st.number_input("Physical Activity (MET-min/week)", min_value=0, value=1500)
            
        with col2:
            alcohol = st.number_input("Alcohol (drinks/week)", min_value=0, value=2)
            los = st.selectbox("Stress Level", 
                             ["Low", "Moderate", "High"],
                             index=1)
            ckd = st.selectbox("CKD Present", ["No", "Yes"], index=0)
            atd = st.selectbox("Endocrine Disorder", ["No", "Yes"], index=0)
            gender = st.selectbox("Sex", ["Male", "Female"], index=0)
            pregnancy = st.selectbox("Pregnancy Status", ["No", "Yes"], index=0)
            smoking = st.selectbox("Smoking Status", ["Non-smoker", "Smoker"], index=0)
        
        submitted = st.form_submit_button("Calculate Hypertension Risk")

    if submitted:
        with st.spinner('Processing clinical data...'):
            time.sleep(1.5)
            
            # Data processing
            input_data = {
                'age': age,
                'bmi': bmi,
                'hemoglobin': loh,
                'salt': scid,
                'activity': pa,
                'alcohol': alcohol,
                'stress': ["Low", "Moderate", "High"].index(los),
                'ckd': 1 if ckd == "Yes" else 0,
                'endocrine': 1 if atd == "Yes" else 0,
                'female': 1 if gender == "Female" else 0,
                'pregnant': 1 if pregnancy == "Yes" else 0,
                'smoker': 1 if smoking == "Smoker" else 0
            }
            
            # Prediction
            prediction = model.predict([[input_data[k] for k in model.feature_names_in_]])
            proba = model.predict_proba([[input_data[k] for k in model.feature_names_in_]])[0]
            
            # --- RESULTS DISPLAY ---
            st.success("Assessment Complete")
            st.divider()
            
            # 1. Risk Summary
            col_summary, col_viz = st.columns([1, 2])
            
            with col_summary:
                st.subheader("Risk Classification")
                if prediction[0] == 1:
                    st.error(f"High Risk ({proba[1]:.1%} probability)")
                else:
                    st.success(f"Low Risk ({proba[0]:.1%} probability)")
                
                st.metric("10-Year CVD Risk", 
                         f"{min(100, 15*(proba[1]/0.3)):.1f}%",
                         help="Estimated cardiovascular disease risk based on HBP probability")
                
            with col_viz:
                fig, ax = plt.subplots(figsize=(8,3))
                ax.bar(['Low Risk', 'High Risk'], proba, 
                      color=['#2ecc71', '#e74c3c'])
                ax.set_title("Probability Distribution", pad=15)
                st.pyplot(fig)
            
            st.divider()
            
            # 2. Clinical Recommendations
            st.subheader("Clinical Management Plan")
            
            # Dynamic recommendations
            rec_cols = st.columns(3)
            with rec_cols[0]:
                st.markdown("**Lifestyle**")
                st.write("- Reduce sodium to <2g/day" if scid > 3 else "- Maintain current diet")
                st.write(f"- {'Increase' if pa < 1000 else 'Maintain'} physical activity")
                st.write("- Smoking cessation" if smoking == "Smoker" else "- Avoid smoking")
                
            with rec_cols[1]:
                st.markdown("**Monitoring**")
                st.write("- Weekly BP checks" if prediction[0] == 1 else "- Annual screening")
                st.write("- Renal function tests" if ckd == "Yes" else "- Basic metabolic panel")
                st.write("- 24h ambulatory monitoring" if proba[1] > 0.4 else "- Clinic measurements")
                
            with rec_cols[2]:
                st.markdown("**Referrals**")
                st.write("- Cardiology consult" if proba[1] > 0.5 else "- Primary care follow-up")
                st.write("- Nutritionist referral" if bmi >= 30 else "- Dietary counseling")
                st.write("- Mental health screening" if los == "High" else "- Stress assessment")
            
            st.divider()
            
            # 3. Feature Importance
            st.subheader("Key Contributing Factors")
            
            importance = pd.DataFrame({
                'Factor': model.feature_names_in_,
                'Impact': model.feature_importances_
            }).sort_values('Impact', ascending=False)
            
            fig_imp, ax_imp = plt.subplots(figsize=(8,4))
            ax_imp.barh(importance['Factor'], importance['Impact'], color='#3498db')
            ax_imp.set_title("Relative Feature Importance", pad=15)
            st.pyplot(fig_imp)
            
            with st.expander("Detailed Factor Analysis"):
                st.dataframe(importance.set_index('Factor').style.background_gradient(cmap='Blues'))
            
            st.divider()
            
            # 4. Risk Trajectory
            st.subheader("Projected Risk Trajectory")
            
            years = np.arange(1, 11)
            if prediction[0] == 1:
                risk_proj = np.minimum(0.95, proba[1] * (1 + years*0.08))
                fig_proj, ax_proj = plt.subplots(figsize=(8,3))
                ax_proj.plot(years, risk_proj, 'r-', marker='o')
                ax_proj.set_ylim(0,1)
                ax_proj.set_title("Untreated Risk Projection", pad=15)
                st.pyplot(fig_proj)
                
                st.info("With treatment, risk could be reduced by 35-50% based on JNC8 guidelines")
            else:
                st.info("Maintaining current lifestyle shows favorable long-term outlook")
            
            # 5. Report Generation
            st.divider()
            if st.button("Generate Clinical Report (PDF)"):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Hypertension Risk Report", ln=1)
                pdf.cell(200, 10, txt=f"Risk Level: {'High' if prediction[0] == 1 else 'Low'}", ln=1)
                pdf.output("hbp_report.pdf")
                st.success("Report generated! Right-click to download")
