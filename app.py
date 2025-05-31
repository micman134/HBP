import streamlit as st
import joblib
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Hypertension Prediction Tool",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS to match your HTML styling
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: transparent !important;
        color: white !important;
        border: 1px solid #6c757d !important;
    }
    .stNumberInput input:focus, .stTextInput input:focus, .stSelectbox select:focus {
        border-color: #007bff !important;
        box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25) !important;
    }
    .stButton>button {
        border: 1px solid #007bff;
        color: #007bff;
        background-color: transparent;
    }
    .stButton>button:hover {
        background-color: #007bff;
        color: white;
    }
    .result-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
    }
    .result-table th, .result-table td {
        border: 1px solid #6c757d;
        padding: 8px;
        text-align: left;
    }
    .result-table th {
        background-color: #343a40;
    }
    .badge {
        padding: 5px 10px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-success {
        background-color: #28a745;
        color: white;
    }
    .badge-warning {
        background-color: #ffc107;
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)

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

# Main app
st.title("Hypertension Predictions")

# Initialize session state for results
if 'result' not in st.session_state:
    st.session_state.result = None
if 'input_values' not in st.session_state:
    st.session_state.input_values = None

# Create form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (yrs)", min_value=0, max_value=120, value=45)
        bmi = st.number_input("Body Mass Index (kg/m²)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
        loh = st.number_input("Level Of Haemoglobin (Hb g/dl)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        gpc = st.number_input("Inbreed Coefficient", min_value=0.0, max_value=1.0, value=0.0, step=0.01, 
                            help="1/2 siblings or cousins, i = loop, Fca = parent GPC")
        pa = st.number_input("Physical Activity (CAL/4.18Kj)", min_value=0.0, max_value=5000.0, value=2000.0, step=100.0)
        
    with col2:
        scid = st.number_input("Salt Content in diet (grams)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
        alcohol = st.number_input("Alcohol Consumption Per Day (millilitres)", min_value=0.0, max_value=1000.0, value=0.0, step=10.0)
        los = st.selectbox("Level Of Stress", ["Select", "Acute/normal stress (1)", "Episodic acute stress (2)", "Chronic Stress (3)"])
        ckd = st.selectbox("Chronic Kidney Disease", ["Select", "Yes (1)", "No (0)"])
        atd = st.selectbox("Adrenal and Thyroid Disorders", ["Select", "Yes (1)", "No (0)"])
        gender = st.selectbox("Gender", ["Select Gender", "Female (1)", "Male (0)"])
        pregnancy = st.selectbox("Pregnancy Status", ["Select", "Yes (1)", "No (0)"])
        smoking = st.selectbox("Smoking Status", ["Select", "Yes (1)", "No (0)"])
    
    submitted = st.form_submit_button("Predict")

# Process form submission
if submitted:
    # Validate all fields
    if (los == "Select" or ckd == "Select" or atd == "Select" or 
        gender == "Select Gender" or pregnancy == "Select" or smoking == "Select"):
        st.error("Please fill in all fields")
    else:
        # Extract numerical values from selects
        los_value = int(los.split("(")[1].replace(")", "")) if los != "Select" else 0
        ckd_value = int(ckd.split("(")[1].replace(")", "")) if ckd != "Select" else 0
        atd_value = int(atd.split("(")[1].replace(")", "")) if atd != "Select" else 0
        gender_value = int(gender.split("(")[1].replace(")", "")) if gender != "Select Gender" else 0
        pregnancy_value = int(pregnancy.split("(")[1].replace(")", "")) if pregnancy != "Select" else 0
        smoking_value = int(smoking.split("(")[1].replace(")", "")) if smoking != "Select" else 0
        
        # Prepare input data in correct order for model
        input_data = [
            loh, gpc, age, bmi, gender_value, pregnancy_value, 
            smoking_value, pa, scid, alcohol, los_value, ckd_value, atd_value
        ]
        
        # Make prediction
        try:
            prediction = model.predict([input_data])
            st.session_state.result = prediction[0]
            st.session_state.input_values = {
                "Age": age,
                "BMI": bmi,
                "Gender": "Female" if gender_value == 1 else "Male",
                "Pregnancy Status": "Yes" if pregnancy_value == 1 else "No",
                "Smoking Status": "Yes" if smoking_value == 1 else "No",
                "Level Of Stress": los.split("(")[0].strip(),
                "Chronic Kidney Disease": "Yes" if ckd_value == 1 else "No",
                "Adrenal and Thyroid Disorders": "Yes" if atd_value == 1 else "No",
                "Geometric Pedigree Coefficient": gpc,
                "Alcohol Consumption Per Day": alcohol,
                "Physical Activity": pa,
                "Salt Content in Diet": scid,
                "Level Of Haemoglobin": loh
            }
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Display results if available
if st.session_state.result is not None:
    st.markdown("---")
    st.header("Prediction Result")
    
    # Result alert
    if st.session_state.result == 0:
        st.success("Negative for High Blood Pressure")
    else:
        st.warning("Positive for High Blood Pressure")
    
    # Display input values in a table
    st.markdown("### Input Values")
    html_table = """
    <table class="result-table">
        <thead>
            <tr>
                <th>S/N</th>
                <th>Attributes</th>
                <th>Values</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for i, (key, value) in enumerate(st.session_state.input_values.items(), 1):
        html_table += f"""
            <tr>
                <td>{i}</td>
                <td>{key}</td>
                <td>{value}</td>
            </tr>
        """
    
    # Add final result row
    result_text = "Negative (-ve)" if st.session_state.result == 0 else "Positive (+ve)"
    badge_class = "badge-success" if st.session_state.result == 0 else "badge-warning"
    html_table += f"""
        <tr>
            <td colspan="2"><b>Result:</b></td>
            <td><span class="badge {badge_class}">{result_text}</span></td>
        </tr>
        <tr>
            <td colspan="3">
                <button onclick="window.print()" class="btn btn-outline-primary">Print Results</button>
            </td>
        </tr>
    """
    
    html_table += """
        </tbody>
    </table>
    """
    
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Add JavaScript for printing
    st.markdown("""
    <script>
    function printResults() {
        window.print();
    }
    </script>
    """, unsafe_allow_html=True)
