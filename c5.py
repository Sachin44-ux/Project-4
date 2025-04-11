import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Medical Prediction Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Navigation
page = st.sidebar.selectbox(
    "Select Prediction Module",
    ["Heart Disease", "Liver Disease", "Diabetes"],
    help="Choose a medical condition to analyze"
)

# Common Explanation Functions
def show_shap_explanation(model, data, feature_names):
    """Generate and display SHAP explanation plots."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data)
    except Exception:
        background = shap.sample(data, 10)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(data)

    fig, ax = plt.subplots(figsize=(10, 6))
    if isinstance(shap_values, list):  # Binary classification
        shap.summary_plot(shap_values[1], data, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, data, feature_names=feature_names, show=False)
    st.pyplot(fig)

def show_lime_explanation(model, scaler, data_point, feature_names, class_names, numeric_columns=None):
    """Generate and return LIME explanation in HTML format."""
    training_data = scaler.transform(np.random.rand(100, len(feature_names)))
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=training_data,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    exp = explainer.explain_instance(data_point, model.predict_proba, num_features=len(feature_names))
    return exp.as_html()

# Heart Disease Prediction Module
if page == "Heart Disease":
    # Load Models and Scaler
    with open(r"D:\project\scaler_heart.sav", 'rb') as f:
        heart_scaler = pickle.load(f)
    with open(r"D:\project\gradient_heart_boosting_model.sav", 'rb') as f:
        heart_model = pickle.load(f)
    with open(r"D:\project\stacking_heart_model.sav", 'rb') as f:
        heart_stacking_model = pickle.load(f)

    # Define Features
    heart_features = [
        'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Exercise Habits',
        'Smoking', 'Family Heart Disease', 'Diabetes', 'BMI', 'High Blood Pressure',
        'Low HDL Cholesterol', 'High LDL Cholesterol', 'Stress Level', 'Sleep Hours',
        'Sugar Consumption', 'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level',
        'Homocysteine Level'
    ]
    numeric_columns = [
        'Age', 'Blood Pressure', 'Cholesterol Level', 'BMI', 'Sleep Hours',
        'Triglyceride Level', 'Fasting Blood Sugar', 'CRP Level', 'Homocysteine Level'
    ]

    # UI Layout
    st.title("❤️ Heart Disease Prediction")
    st.write("Enter patient details below to assess heart disease risk")

    # Input Collection
    inputs = {}
    col1, col2 = st.columns(2)

    with col1:
        inputs['Age'] = st.number_input("Age", 1, 120, 40, key='heart_age')
        inputs['Blood Pressure'] = st.number_input("Blood Pressure (mmHg)", 50, 200, 120, key='heart_bp')
        inputs['Cholesterol Level'] = st.number_input("Cholesterol (mg/dl)", 100, 400, 200, key='heart_chol')
        inputs['BMI'] = st.number_input("BMI", 10.0, 50.0, 25.0, 0.1, key='heart_bmi')
        inputs['Sleep Hours'] = st.number_input("Sleep Hours/Night", 1.0, 12.0, 7.0, 0.1, key='heart_sleep')

    with col2:
        inputs['Triglyceride Level'] = st.number_input("Triglycerides (mg/dl)", 30, 500, 150, key='heart_trig')
        inputs['Fasting Blood Sugar'] = st.number_input("Fasting Glucose (mg/dl)", 50, 300, 90, key='heart_fbs')
        inputs['CRP Level'] = st.number_input("CRP Level (mg/L)", 0.0, 20.0, 1.0, 0.1, key='heart_crp')
        inputs['Homocysteine Level'] = st.number_input("Homocysteine (µmol/L)", 2.0, 50.0, 10.0, 0.1, key='heart_hcy')

    # Categorical Inputs
    st.subheader("Health Factors")
    col3, col4 = st.columns(2)
    with col3:
        inputs['Gender'] = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key='heart_gender')
        inputs['Exercise Habits'] = st.selectbox("Exercise", [0, 1], format_func=lambda x: "None" if x == 0 else "Regular", key='heart_exercise')
        inputs['Smoking'] = st.selectbox("Smoking", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='heart_smoking')
        inputs['Family Heart Disease'] = st.selectbox("Family History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='heart_family')
        inputs['Diabetes'] = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='heart_diabetes')

    with col4:
        inputs['High Blood Pressure'] = st.selectbox("High BP", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='heart_bp_cat')
        inputs['Low HDL Cholesterol'] = st.selectbox("Low HDL", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='heart_hdl')
        inputs['High LDL Cholesterol'] = st.selectbox("High LDL", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='heart_ldl')
        inputs['Stress Level'] = st.selectbox("Stress", [0, 1], format_func=lambda x: "Low" if x == 0 else "High", key='heart_stress')
        inputs['Sugar Consumption'] = st.selectbox("Sugar Intake", [0, 1], format_func=lambda x: "Normal" if x == 0 else "High", key='heart_sugar')

    # Prediction Logic
    if st.button("Predict Heart Disease", key='heart_predict'):
        df = pd.DataFrame([inputs], columns=heart_features)
        scaled = heart_scaler.transform(df)
        proba = heart_model.predict_proba(scaled)[0]
        prediction = np.argmax(proba)

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"High Risk Detected ({proba[1]:.2%} probability)")
        else:
            st.success(f"Low Risk Detected ({proba[0]:.2%} probability)")

        # Explanations
        col5, col6 = st.columns(2)
        with col5:
            st.subheader("SHAP Feature Importance")
            show_shap_explanation(heart_model, scaled, heart_features)
        with col6:
            st.subheader("LIME Local Explanation")
            lime_html = show_lime_explanation(
                heart_stacking_model, heart_scaler, scaled[0],
                heart_features, ['Disease', 'No Disease']
            )
            st.components.v1.html(lime_html, height=600)

# Liver Disease Prediction Module
elif page == "Liver Disease":
    # Load Models and Scaler
    with open(r"D:\project\_liver_scaler.sav", 'rb') as f:
        liver_scaler = pickle.load(f)
    with open(r"D:\project\gradient_liver_boosting_model.sav", 'rb') as f:
        liver_model = pickle.load(f)
    with open(r"D:\project\stacking_liver_model.sav", 'rb') as f:
        liver_stacking_model = pickle.load(f)

    liver_features = [
        "Age", "Gender", "BMI", "AlcoholConsumption", "Smoking",
        "GeneticRisk", "PhysicalActivity", "Diabetes", "Hypertension",
        "LiverFunctionTest"
    ]

    # UI Layout
    st.title("🩺 Liver Disease Prediction")
    st.write("Enter patient details below to assess liver disease risk")

    inputs = {}
    col1, col2 = st.columns(2)
    with col1:
        inputs["Age"] = st.number_input("Age", 0, 120, 40, key='liver_age')
        inputs["Gender"] = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key='liver_gender')
        inputs["BMI"] = st.number_input("BMI", 10.0, 50.0, 25.0, 0.1, key='liver_bmi')
        inputs["AlcoholConsumption"] = st.slider("Alcohol Drinks/Week", 0, 50, 0, key='liver_alcohol')
        inputs["Smoking"] = st.selectbox("Smoker", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='liver_smoke')

    with col2:
        inputs["GeneticRisk"] = st.slider("Genetic Risk (0-1)", 0.0, 1.0, 0.5, 0.01, key='liver_genetic')
        inputs["PhysicalActivity"] = st.slider("Exercise Hours/Week", 0, 20, 5, key='liver_exercise')
        inputs["Diabetes"] = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='liver_diabetes')
        inputs["Hypertension"] = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='liver_htn')
        inputs["LiverFunctionTest"] = st.number_input("Liver Test Score", 0.0, 100.0, 50.0, key='liver_test')

    if st.button("Predict Liver Disease", key='liver_predict'):
        df = pd.DataFrame([inputs], columns=liver_features)
        scaled = liver_scaler.transform(df)
        proba = liver_model.predict_proba(scaled)[0]
        prediction = np.argmax(proba)

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"High Risk Detected ({proba[1]:.2%} probability)")
        else:
            st.success(f"Low Risk Detected ({proba[0]:.2%} probability)")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("SHAP Feature Importance")
            show_shap_explanation(liver_model, scaled, liver_features)
        with col2:
            st.subheader("LIME Local Explanation")
            lime_html = show_lime_explanation(
                liver_stacking_model, liver_scaler, scaled[0],
                liver_features, ['No Disease', 'Disease']
            )
            st.components.v1.html(lime_html, height=600)

elif page == "Diabetes":
    # Load Models and Scaler
    with open(r"D:\project\_diabetes_scaler (1).sav", 'rb') as f:
        diabetes_scaler = pickle.load(f)
    with open(r"D:\project\gradient_diabetes_boosting_model (1).sav", 'rb') as f:
        diabetes_model = pickle.load(f)
    with open(r"D:\project\stacking_diabetes_model (1).sav", 'rb') as f:
        diabetes_stacking_model = pickle.load(f)

    # Define Features in the original order from diabetes_dataset00.csv
    diabetes_features = [
        'Genetic Markers', 'Autoantibodies', 'Family History', 'Environmental Factors',
        'Insulin Levels', 'Age', 'BMI', 'Physical Activity', 'Dietary Habits',
        'Blood Pressure', 'Cholesterol Levels', 'Waist Circumference', 'Blood Glucose Levels',
        'Ethnicity', 'Socioeconomic Factors', 'Smoking Status', 'Alcohol Consumption',
        'Glucose Tolerance Test', 'History of PCOS', 'Previous Gestational Diabetes',
        'Pregnancy History', 'Weight Gain During Pregnancy', 'Pancreatic Health',
        'Pulmonary Function', 'Cystic Fibrosis Diagnosis', 'Steroid Use History',
        'Genetic Testing', 'Neurological Assessments', 'Liver Function Tests',
        'Digestive Enzyme Levels', 'Urine Test', 'Birth Weight', 'Early Onset Symptoms'
    ]
    original_columns = ['Target'] + diabetes_features

    class_names = [
        'MODY', 'Secondary Diabetes', 'Cystic Fibrosis-Related Diabetes (CFRD)',
        'Type 1 Diabetes', 'Neonatal Diabetes Mellitus (NDM)', 'Wolcott-Rallison Syndrome',
        'Type 2 Diabetes', 'Prediabetic', 'Gestational Diabetes',
        'Type 3c Diabetes (Pancreatogenic Diabetes)', 'Wolfram Syndrome',
        'Steroid-Induced Diabetes', 'LADA'
    ]

    # UI Layout (unchanged)
    st.title("🩺 Diabetes Prediction")
    st.write("Enter patient details below to assess diabetes type risk")

    # Input Collection (unchanged)
    inputs = {}
    st.subheader("Patient Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        inputs['Genetic Markers'] = st.selectbox("Genetic Markers", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_genetic_markers')
        inputs['Autoantibodies'] = st.selectbox("Autoantibodies", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_autoantibodies')
        inputs['Family History'] = st.selectbox("Family History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_family_history')
        inputs['Environmental Factors'] = st.selectbox("Environmental Factors", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_env_factors')
        inputs['Insulin Levels'] = st.number_input("Insulin Levels (µU/mL)", 0.0, 300.0, 15.0, 0.1, key='diab_insulin')
        inputs['Age'] = st.number_input("Age", 0, 120, 40, key='diab_age')
        inputs['BMI'] = st.number_input("BMI", 10.0, 50.0, 25.0, 0.1, key='diab_bmi')
        inputs['Physical Activity'] = st.slider("Physical Activity (hours/week)", 0, 20, 5, key='diab_activity')
        inputs['Dietary Habits'] = st.selectbox("Unhealthy Dietary Habits", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_diet')
        inputs['Blood Pressure'] = st.number_input("Blood Pressure (mmHg)", 50, 200, 120, key='diab_bp')
        inputs['Cholesterol Levels'] = st.number_input("Cholesterol (mg/dL)", 100, 400, 200, key='diab_chol')

    with col2:
        inputs['Waist Circumference'] = st.number_input("Waist Circumference (cm)", 50, 150, 80, key='diab_waist')
        inputs['Blood Glucose Levels'] = st.number_input("Blood Glucose (mg/dL)", 50, 300, 100, key='diab_glucose')
        inputs['Ethnicity'] = st.selectbox("High-Risk Ethnicity", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_ethnicity')
        inputs['Socioeconomic Factors'] = st.selectbox("Low Socioeconomic Status", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_socio')
        inputs['Smoking Status'] = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_smoking')
        inputs['Alcohol Consumption'] = st.slider("Alcohol Drinks/Week", 0, 50, 0, key='diab_alcohol')
        inputs['Glucose Tolerance Test'] = st.number_input("Glucose Tolerance (mg/dL)", 50, 300, 140, key='diab_gtt')
        inputs['History of PCOS'] = st.selectbox("History of PCOS", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_pcos')
        inputs['Previous Gestational Diabetes'] = st.selectbox("Previous Gestational Diabetes", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_gdm')
        inputs['Pregnancy History'] = st.selectbox("Pregnancy History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_pregnancy')
        inputs['Weight Gain During Pregnancy'] = st.number_input("Weight Gain During Pregnancy (kg)", 0.0, 50.0, 0.0, 0.1, key='diab_weight_gain')

    with col3:
        inputs['Pancreatic Health'] = st.selectbox("Poor Pancreatic Health", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_pancreas')
        inputs['Pulmonary Function'] = st.selectbox("Impaired Pulmonary Function", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_pulmonary')
        inputs['Cystic Fibrosis Diagnosis'] = st.selectbox("Cystic Fibrosis Diagnosis", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_cf')
        inputs['Steroid Use History'] = st.selectbox("Steroid Use History", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_steroid')
        inputs['Genetic Testing'] = st.selectbox("Positive Genetic Testing", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_gen_test')
        inputs['Neurological Assessments'] = st.selectbox("Abnormal Neurological Assessment", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_neuro')
        inputs['Liver Function Tests'] = st.number_input("Liver Function Test Score", 0.0, 100.0, 50.0, key='diab_liver')
        inputs['Digestive Enzyme Levels'] = st.number_input("Digestive Enzyme Levels (U/L)", 0.0, 200.0, 100.0, key='diab_enzyme')
        inputs['Urine Test'] = st.selectbox("Abnormal Urine Test", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_urine')
        inputs['Birth Weight'] = st.number_input("Birth Weight (kg)", 0.5, 6.0, 3.0, 0.1, key='diab_birth_weight')
        inputs['Early Onset Symptoms'] = st.selectbox("Early Onset Symptoms", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key='diab_early')

    if st.button("Predict Diabetes Type", key='diab_predict'):
    # Add dummy Target column
        inputs['Target'] = 0
    df = pd.DataFrame([inputs], columns=original_columns)
    scaled = diabetes_scaler.transform(df)
    scaled_features = scaled[:, 1:]  # Remove Target column

    # Predict
    proba = diabetes_stacking_model.predict_proba(scaled_features)[0]
    prediction = np.argmax(proba)
    predicted_class = class_names[prediction]
    predicted_proba = proba[prediction]

    # Display Result
    st.subheader("Prediction Result")
    st.warning(f"Predicted Diabetes Type: **{predicted_class}** ({predicted_proba:.2%} probability)")
    proba_df = pd.DataFrame({'Diabetes Type': class_names, 'Probability': proba})
    st.bar_chart(proba_df.set_index('Diabetes Type'))

    # Updated SHAP Explanation
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("SHAP Explanation")
        
        # Generate background data
        n_samples = 100
        synthetic_data = np.random.rand(n_samples, len(diabetes_features))
        synthetic_with_target = np.hstack([np.zeros((n_samples, 1)), synthetic_data])
        scaled_background = diabetes_scaler.transform(synthetic_with_target)[:, 1:]

        # Create SHAP explainer for multi-class
        explainer = shap.KernelExplainer(
            diabetes_stacking_model.predict_proba,
            scaled_background
        )
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(scaled_features[0:1])  # Use 2D array
        
        # For multi-class, shap_values is a list of arrays (one per class)
        if isinstance(shap_values, list):
            shap_values_class = shap_values[prediction]  # SHAP values for predicted class
        else:
            shap_values_class = shap_values  # Fallback for unexpected shape

        # Plot 1: Bar plot for feature importance
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_class,
            scaled_features,
            feature_names=diabetes_features,
            plot_type="bar",
            max_display=len(diabetes_features),
            show=False
        )
        plt.title(f"SHAP Feature Importance for {predicted_class}")
        st.pyplot(plt.gcf())
        plt.clf()

        # Plot 2: Beeswarm plot for detailed view
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_class,
            scaled_features,
            feature_names=diabetes_features,
            max_display=len(diabetes_features),
            show=False
        )
        plt.title(f"SHAP Values Distribution for {predicted_class}")
        st.pyplot(plt.gcf())
        plt.clf()

    with col2:
        st.subheader("LIME Local Explanation")
        def diabetes_lime_explanation():
            n_features = len(diabetes_features)
            random_data = np.random.rand(100, n_features)
            random_data_with_target = np.hstack([np.zeros((100, 1)), random_data])
            scaled_with_target = diabetes_scaler.transform(random_data_with_target)
            training_data = scaled_with_target[:, 1:]

            explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                feature_names=diabetes_features,
                class_names=class_names,
                mode='classification'
            )
            exp = explainer.explain_instance(
                scaled_features[0],
                diabetes_stacking_model.predict_proba,
                num_features=n_features
            )
            return exp.as_html()

        lime_html = diabetes_lime_explanation()
        st.components.v1.html(lime_html, height=600)

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Disclaimer**: This AI tool provides risk assessments based on input data and is not a substitute
    for professional medical diagnosis. Please consult a healthcare professional for medical advice.
    """
)

