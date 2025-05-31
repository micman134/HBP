import streamlit as st
import joblib

# Load the model
model = joblib.load('model.joblib')

# Set up the Streamlit app
st.title("BPA Risk Prediction Tool")

# Create input form
with st.form("prediction_form"):
    st.header("Patient Information")
    
    # Split inputs into two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0, step=0.1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        pregnancy = st.selectbox("Pregnancy Status", ["No", "Yes"])
        smoking = st.selectbox("Smoking Status", ["Non-smoker", "Smoker"])
        los = st.selectbox("Level of Stress", ["Low", "Medium", "High"])
    
    with col2:
        ckd = st.selectbox("Chronic Kidney Disease", ["No", "Yes"])
        atd = st.selectbox("Antidepressant Therapy", ["No", "Yes"])
        gpc = st.selectbox("General Physical Condition", ["Poor", "Fair", "Good", "Excellent"])
        alcohol = st.selectbox("Alcohol Consumption", ["None", "Light", "Moderate", "Heavy"])
        pa = st.selectbox("Physical Activity", ["Sedentary", "Light", "Moderate", "Active"])
        scid = st.selectbox("SCID Diagnosis", ["No", "Yes"])
        loh = st.selectbox("Level of Happiness", ["Low", "Medium", "High"])
    
    # Submit button
    submitted = st.form_submit_button("Predict BPA Risk")
    
    if submitted:
        # Convert inputs to model format
        gender_encoded = 1 if gender == "Female" else 0
        pregnancy_encoded = 1 if pregnancy == "Yes" else 0
        smoking_encoded = 1 if smoking == "Smoker" else 0
        los_encoded = ["Low", "Medium", "High"].index(los)
        ckd_encoded = 1 if ckd == "Yes" else 0
        atd_encoded = 1 if atd == "Yes" else 0
        gpc_encoded = ["Poor", "Fair", "Good", "Excellent"].index(gpc)
        alcohol_encoded = ["None", "Light", "Moderate", "Heavy"].index(alcohol)
        pa_encoded = ["Sedentary", "Light", "Moderate", "Active"].index(pa)
        scid_encoded = 1 if scid == "Yes" else 0
        loh_encoded = ["Low", "Medium", "High"].index(loh)
        
        # Make prediction
        prediction = model.predict([[loh_encoded, gpc_encoded, age, bmi, gender_encoded, 
                                   pregnancy_encoded, smoking_encoded, pa_encoded, 
                                   scid_encoded, alcohol_encoded, los_encoded, 
                                   ckd_encoded, atd_encoded]])
        
        # Display result
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error("High risk of BPA")
        else:
            st.success("Low risk of BPA")
            
        # Show feature importance if available (for RandomForest)
        if hasattr(model, 'feature_importances_'):
            st.subheader("Feature Importance")
            features = ['Level of Happiness', 'General Physical Condition', 'Age', 'BMI', 
                        'Gender', 'Pregnancy', 'Smoking', 'Physical Activity', 'SCID Diagnosis', 
                        'Alcohol', 'Level of Stress', 'Chronic Kidney Disease', 'Antidepressant Therapy']
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.bar_chart(importance_df.set_index('Feature'))
