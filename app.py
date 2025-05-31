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
    st.image("https://via.placeholder.com/150x50?text=HBP+Tool", width=150)
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
    - **Demographic**: Age, Gender, Pregnancy Status
    - **Lifestyle**: Smoking, Alcohol Consumption, Physical Activity
    - **Clinical**: Chronic Kidney Disease, Thyroid Disorders
    - **Biochemical**: Hemoglobin Levels, Salt Intake
    - **Genetic**: Inbreeding Coefficient
    """)
    
    try:
        st.image("ontology.PNG",
                caption="HBP Risk Factor Ontology Diagram",
                use_column_width=True)
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

# Main Prediction Page
else:
    st.title("High Blood Pressure Risk Prediction Tool")
    st.write("""
    This tool predicts the risk of HBP based on patient characteristics.
    Please fill in all the fields below and click 'Predict'.
    """)

    # Initialize session state for form submission
    if 'submitted' not in st.session_state:
        st.session_state.submitted = False

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
        
        submitted = st.form_submit_button("Predict HBP Risk")
        if submitted:
            st.session_state.submitted = True

    # Prediction logic - only runs if form was submitted
    if st.session_state.get('submitted', False):
        # Validate all fields are selected
        if (los == "Select" or ckd == "Select" or atd == "Select" or 
            gender == "Select Gender" or pregnancy == "Select" or smoking == "Select"):
            st.error("Please fill in all fields before submitting")
            st.session_state.submitted = False
        else:
            try:
                # Show spinner while processing
                with st.spinner('Analyzing health data and calculating risk...'):
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
                    
                    prediction = model.predict(input_values)
                    proba = model.predict_proba(input_values)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
                
                # Display results in two columns (charts on left)
                col_results, col_charts = st.columns([1, 1.5])
                
                with col_results:
                    st.subheader("Prediction Result")
                    if prediction[0] == 1:
                        st.error("**High risk of HBP**")
                        st.warning("This patient shows characteristics associated with higher HBP risk. Consider additional screening.")
                    else:
                        st.success("**Low risk of HBP**")
                        st.info("This patient shows characteristics associated with lower HBP risk.")
                    
                    st.markdown("---")
                    st.write(f"**Probability of low risk:** {proba[0]:.1%}")
                    st.write(f"**Probability of high risk:** {proba[1]:.1%}")
                    st.progress(proba[1])
                
                with col_charts:
                    # Probability chart (left side)
                    st.subheader("Risk Probability")
                    fig1, ax1 = plt.subplots(figsize=(7, 4))
                    ax1.bar(['Low Risk', 'High Risk'], proba, color=['green', 'red'])
                    ax1.set_ylim(0, 1)
                    ax1.set_ylabel('Probability')
                    for i, v in enumerate(proba):
                        ax1.text(i, v + 0.02, f"{v:.1%}", ha='center')
                    st.pyplot(fig1)
                    
                    # Feature importance (left side)
                    if hasattr(model, "feature_importances_"):
                        st.subheader("Feature Importance")
                        importance = model.feature_importances_
                        fi_df = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values('Importance', ascending=False)
                        
                        fig2, ax2 = plt.subplots(figsize=(7, 5))
                        ax2.barh(fi_df['Feature'], fi_df['Importance'], color='skyblue')
                        ax2.set_xlabel('Importance Score')
                        st.pyplot(fig2)
                        
                        with st.expander("View Feature Importance Table"):
                            st.dataframe(fi_df.set_index('Feature').style.format({'Importance': '{:.3f}'}))
                
                # PDF Report Generation (Fixed)
                def create_pdf():
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    
                    # Header
                    pdf.cell(200, 10, txt="Hypertension Risk Report", ln=1, align='C')
                    pdf.ln(10)
                    
                    # Patient Info
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(200, 10, txt="Patient Summary", ln=1)
                    pdf.set_font("Arial", '', 12)
                    pdf.cell(200, 10, txt=f"Age: {age} | BMI: {bmi:.1f}", ln=1)
                    pdf.cell(200, 10, txt=f"Gender: {gender} | Pregnant: {pregnancy}", ln=1)
                    pdf.ln(5)
                    
                    # Risk Assessment
                    pdf.set_font("Arial", 'B', 12)
                    risk_level = "High" if prediction[0] == 1 else "Low"
                    pdf.cell(200, 10, txt=f"Risk Level: {risk_level} ({proba[1]:.1%} probability)", ln=1)
                    pdf.ln(5)
                    
                    # Recommendations
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(200, 10, txt="Clinical Recommendations:", ln=1)
                    pdf.set_font("Arial", '', 12)
                    
                    recs = []
                    if prediction[0] == 1:
                        recs.append("Immediate clinical evaluation")
                        recs.append("Consider 24-hour BP monitoring")
                    if bmi >= 30:
                        recs.append("Weight management program")
                    if scid > 3:
                        recs.append("Sodium restriction diet")
                    
                    for rec in recs:
                        pdf.cell(200, 10, txt=f"- {rec}", ln=1)
                    
                    return pdf
                
                # Generate and download PDF
                pdf = create_pdf()
                st.download_button(
                    label="Download PDF Report",
                    data=pdf.output(dest='S').encode('latin1'),
                    file_name="hbp_risk_report.pdf",
                    mime="application/pdf"
                )
                
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
