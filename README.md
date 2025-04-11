# Project-4
Multiple disease prediction
.
---


## Overview

The **Medical Prediction Suite** is an advanced, Streamlit-based web application designed to predict the risk of three medical conditions: **Heart Disease**, **Liver Disease**, and **Diabetes**. Built using Python and leveraging state-of-the-art machine learning techniques, this suite employs ensemble models (Gradient Boosting, AdaBoost, CatBoost) and a stacking classifier with a Random Forest meta-learner. It enhances interpretability with SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) visualizations, making it a valuable tool for understanding feature contributions to predictions.

### Objectives
- Provide accurate risk assessments for heart disease, liver disease, and diabetes.
- Offer interpretable insights into model predictions for both global (feature importance) and local (individual prediction) contexts.
- Create an intuitive, user-friendly interface for inputting patient data and visualizing results.

### Key Features
- **Heart Disease Prediction**: Assesses binary risk (disease/no disease) based on 19 features, including age, cholesterol, blood pressure, and lifestyle factors.
- **Liver Disease Prediction**: Predicts binary risk using 10 features like BMI, alcohol consumption, and liver function tests.
- **Diabetes Prediction**: Performs multi-class classification across 13 diabetes types (e.g., Type 1, Type 2, Gestational) using 33 features.
- **Model Interpretability**:
  - **SHAP**: Visualizes global feature importance and value distributions.
  - **LIME**: Provides local explanations for individual predictions in HTML format.
- **Interactive UI**: Built with Streamlit, featuring a sidebar for module selection, input forms, and result displays.

## Installation

### Prerequisites
- **Operating System**: Windows, macOS, or Linux
- **Python**: Version 3.8 or higher
- **Dependencies**: Listed in `requirements.txt` (create this file if not provided)
- **Disk Space**: At least 500 MB for models, datasets, and dependencies
- **Internet**: Required for initial package installation

### Dependencies
The project relies on the following Python libraries:
- `streamlit` - Web app framework
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scikit-learn` - Machine learning utilities
- `catboost` - Gradient boosting library
- `xgboost` - Extreme gradient boosting
- `shap` - SHAP explanations
- `lime` - LIME explanations
- `matplotlib` - Plotting library
- `pickle` - Model serialization

### Installation Steps
1. **Clone the Repository** (if hosted on a VCS like GitHub):
   ```bash
   git clone <repository-url>
   cd medical-prediction-suite
   ```
   If not hosted, copy the project folder to your local machine.

2. **Set Up a Virtual Environment** (recommended to avoid conflicts):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   If a `requirements.txt` file exists:
   ```bash
   pip install -r requirements.txt
   ```
   Otherwise, install manually:
   ```bash
   pip install streamlit pandas numpy scikit-learn catboost xgboost shap lime matplotlib
   ```

4. **Verify Installation**:
   Run a test command to ensure Streamlit works:
   ```bash
   streamlit hello
   ```
   This should open a demo page in your browser.

5. **Prepare Pre-trained Models**:
   The application requires pre-trained model files saved in `.sav` format (serialized with `pickle`). Place these files in `D:\project\` (or update paths in the code):
   - **Heart Disease**:
     - `scaler_heart.sav`
     - `gradient_heart_boosting_model.sav`
     - `stacking_heart_model.sav`
   - **Liver Disease**:
     - `_liver_scaler.sav`
     - `gradient_liver_boosting_model.sav`
     - `stacking_liver_model.sav`
   - **Diabetes**:
     - `_diabetes_scaler (1).sav`
     - `gradient_diabetes_boosting_model (1).sav`
     - `stacking_diabetes_model (1).sav`

   If you don’t have these files, train the models using the provided Jupyter notebooks (see [Model Training](#model-training)).

## Usage

### Running the Application
1. **Launch the App**:
   Navigate to the project directory and run:
   ```bash
   streamlit run medical_prediction_suite.py
   ```
   Replace `medical_prediction_suite.py` with the actual filename of the main script.

2. **Access the Interface**:
   - The app opens in your default browser (e.g., `http://localhost:8501`).
   - Use the sidebar to select a prediction module: "Heart Disease," "Liver Disease," or "Diabetes."

### Using the Prediction Modules
Each module has a tailored interface for data input and result display:

#### Heart Disease Prediction
- **Inputs**: 19 features (e.g., Age, Blood Pressure, Cholesterol Level, Smoking, BMI).
  - Numeric: Enter values like Age (1–120), Blood Pressure (50–200 mmHg).
  - Categorical: Select options (e.g., Gender: Male/Female, Smoking: Yes/No).
- **Action**: Click "Predict Heart Disease."
- **Output**:
  - Risk probability (e.g., "High Risk Detected (75.20% probability)").
  - SHAP summary plot (feature importance).
  - LIME HTML explanation (local feature contributions).

#### Liver Disease Prediction
- **Inputs**: 10 features (e.g., Age, BMI, Alcohol Consumption, Liver Function Test).
  - Numeric: sliders or inputs (e.g., Alcohol Drinks/Week: 0–50).
  - Categorical: Select options (e.g., Gender, Smoking).
- **Action**: Click "Predict Liver Disease."
- **Output**:
  - Risk probability (binary: High/Low).
  - SHAP and LIME visualizations.

#### Diabetes Prediction
- **Inputs**: 33 features (e.g., Genetic Markers, Blood Glucose Levels, Pregnancy History).
  - Numeric: Inputs or sliders (e.g., BMI: 10–50, Glucose: 50–300 mg/dL).
  - Categorical: Yes/No options (e.g., Family History, Steroid Use).
- **Action**: Click "Predict Diabetes Type."
- **Output**:
  - Predicted diabetes type (e.g., "Type 2 Diabetes (82.50% probability)").
  - Probability bar chart across 13 classes.
  - SHAP bar and beeswarm plots.
  - LIME HTML explanation.

### Interpreting Results
- **Prediction**: Color-coded (red for high risk, green for low risk, yellow for diabetes type).
- **SHAP**: Bar plots show feature importance; beeswarm plots show value distributions (Diabetes module).
- **LIME**: HTML table highlights features driving the prediction for the specific input.

## Model Training

The models were developed and trained in Google Colab using the following datasets and processes:

### Datasets
1. **Heart Disease**: `heart_statlog_cleveland_hungary_final.csv`
   - Features: 11 (e.g., age, cholesterol, resting bp s).
   - Target: Binary (0 = No Disease, 1 = Disease).
2. **Liver Disease**: `Liver_disease_data.csv`
   - Features: 10 (e.g., Age, BMI, AlcoholConsumption).
   - Target: Binary (0 = No Disease, 1 = Disease).
3. **Diabetes**: `Dataset Diabetes.csv`
   - Features: 33 (e.g., Glucose, BMI, Family History).
   - Target: Multi-class (13 diabetes types).

### Training Process
1. **Preprocessing**:
   - **Scaling**: `StandardScaler` applied to numeric features.
   - **Diabetes**: Oversampling with `RandomOverSampler` to address class imbalance.
2. **Base Models**:
   - Gradient Boosting (`GradientBoostingClassifier`)
   - CatBoost (`CatBoostClassifier`)
   - AdaBoost (`AdaBoostClassifier`)
3. **Stacking**:
   - Meta-learner: `RandomForestClassifier` (n_estimators=100).
   - Configuration: `StackingClassifier` with `passthrough=True`.
4. **Evaluation**:
   - Metrics: Training and testing accuracy.
   - Cross-validation: 5-fold `StratifiedKFold` for robustness.
5. **Explainability**:
   - SHAP: `TreeExplainer` or `KernelExplainer` for base and meta-models.
   - LIME: `LimeTabularExplainer` with synthetic training data.

### Reproducing Training
- Use the provided Jupyter notebooks:
  - `heart.ipynb`
  - `liv2.ipynb`
  - `diabetes.ipynb`
- Update file paths to your dataset locations and run the cells sequentially.

## File Structure
```
medical-prediction-suite/
├── medical_prediction_suite.py    # Main Streamlit app script
├── D:\project\                    # Model storage directory (update as needed)
│   ├── scaler_heart.sav
│   ├── gradient_heart_boosting_model.sav
│   ├── stacking_heart_model.sav
│   ├── _liver_scaler.sav
│   ├── gradient_liver_boosting_model.sav
│   ├── stacking_liver_model.sav
│   ├── _diabetes_scaler (1).sav
│   ├── gradient_diabetes_boosting_model (1).sav
│   ├── stacking_diabetes_model (1).sav
├── heart.ipynb                   # Heart disease training notebook
├── liv2.ipynb                    # Liver disease training notebook
├── diabetes.ipynb                # Diabetes training notebook
├── requirements.txt              # Dependency list (create if needed)
└── README.md                     # This file
```

## Troubleshooting
- **FileNotFoundError**: Ensure model files are in `D:\project\` or update paths in the script.
- **ModuleNotFoundError**: Verify all dependencies are installed (`pip list`).
- **SHAP/LIME Errors**: Check model compatibility; Diabetes multi-class SHAP may need background data adjustments.
- **Streamlit Issues**: Run `streamlit doctor` to diagnose environment problems.

## Limitations
- **Model Dependency**: Requires pre-trained `.sav` files; retraining is needed if missing.
- **Path Hardcoding**: Update file paths in the script if not using `D:\project\`.
- **Scalability**: SHAP and LIME computations may slow down with large datasets.
- **Diabetes Multi-class**: SHAP visualizations may require tuning for clarity across 13 classes.

## Disclaimer
This tool is intended for **educational and research purposes only**. It provides risk assessments based on machine learning models and input data but is **not a substitute for professional medical diagnosis**. Consult a healthcare professional for accurate medical advice.

## Contributing
- **Bug Reports**: Open an issue with details (e.g., error logs, steps to reproduce).
- **Feature Requests**: Suggest enhancements via issues or pull requests.
- **Pull Requests**: Fork the repository, make changes, and submit with a clear description.

## License
This project is licensed under the MIT License—see the `LICENSE` file for details (create one if absent).

## Contact
For questions or support, reach out via [insert contact method, e.g., email or GitHub issues].

---

### Customization Notes
- Replace `<repository-url>` with the actual Git URL if applicable.
- Generate `requirements.txt` with `pip freeze > requirements.txt` after installing dependencies.
- Adjust paths (e.g., `D:\project\`) to match your setup.
- Add a `LICENSE` file if you choose MIT or another license.

This README provides a thorough guide for users and developers. Let me know if you need further refinements!
