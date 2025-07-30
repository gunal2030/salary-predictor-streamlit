import streamlit as st
import pandas as pd
import joblib
import json
import numpy as np # Import numpy for potential NaN handling

# --- Configuration ---
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💵",
    layout="centered"
)

# --- LOAD THE SAVED FILES ---
@st.cache_resource # Cache the model and scaler loading to run only once
def load_model_files():
    try:
        model = joblib.load('salary_model.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('model_columns.json', 'r') as f:
            model_columns = json.load(f)
        return model, scaler, model_columns
    except FileNotFoundError:
        st.error("Error: Model files not found. Please ensure 'salary_model.pkl', 'scaler.pkl', and 'model_columns.json' are in the same directory.")
        st.stop() # Stop the app if essential files are missing
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.stop()

model, scaler, model_columns = load_model_files()

# --- Application Title and Description ---
st.title("Employee Salary Prediction 💵")
st.write("Enter the details below to predict the salary bracket (<=50K or >50K).")

# --- Input Form ---
st.header("Employee Details")

# Create input widgets for each feature
# Numerical Inputs
age = st.number_input("Age", min_value=17, max_value=90, value=38, step=1)
fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=1, value=189000, step=1000)
educational_num = st.number_input("Educational Number", min_value=1, max_value=16, value=9, step=1)
capital_gain = st.number_input("Capital Gain", min_value=0, value=0, step=100)
capital_loss = st.number_input("Capital Loss", min_value=0, value=0, step=100)
hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40, step=1)

# Categorical Inputs (using st.selectbox)
# Ensure options match categories present in your training data
workclass_options = ["Private", "Self-emp-not-inc", "Local-gov", "Federal-gov", "State-gov", "Self-emp-inc", "Without-pay", "Never-worked"]
workclass = st.selectbox("Work Class", workclass_options)

education_options = ["HS-grad", "Some-college", "Bachelors", "Masters", "Assoc-voc", "11th", "Assoc-acdm", "10th", "7th-8th", "Prof-school", "9th", "12th", "Doctorate", "5th-6th", "1st-4th", "Preschool"]
education = st.selectbox("Education Level", education_options)

marital_status_options = ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
marital_status = st.selectbox("Marital Status", marital_status_options)

occupation_options = ["Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical", "Sales", "Other-service", "Machine-op-inspct", "Transport-moving", "Handlers-cleaners", "Farming-fishing", "Tech-support", "Protective-serv", "Priv-house-serv", "Armed-Forces"]
occupation = st.selectbox("Occupation", occupation_options)

relationship_options = ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife", "Other-relative"]
relationship = st.selectbox("Relationship", relationship_options)

race_options = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]
race = st.selectbox("Race", race_options)

gender_options = ["Male", "Female"]
gender = st.selectbox("Gender", gender_options)

native_country_options = ["United-States", "Mexico", "Philippines", "Germany", "Puerto-Rico", "Canada", "India", "El-Salvador", "Cuba", "England", "China", "Jamaica", "South", "Italy", "Dominican-Republic", "Japan", "Guatemala", "Columbia", "Poland", "France", "Haiti", "Portugal", "Taiwan", "Iran", "Nicaragua", "Peru", "Greece", "Ecuador", "Ireland", "Hong", "Cambodia", "Trinadad&Tobago", "Laos", "Thailand", "Yugoslavia", "Outlying-US(Guam-USVI-etc)", "Hungary", "Honduras", "Scotland", "Holand-Netherlands"]
native_country = st.selectbox("Native Country", native_country_options)


# --- Prediction Logic (when button is pressed) ---
if st.button("Predict Salary"):
    # Create a dictionary from the collected inputs
    input_data = {
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'education': education,
        'educational-num': educational_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encode the input data
    # Ensure all columns from model_columns are present, fill with 0 if missing
    input_encoded = pd.get_dummies(input_df)

    # Align columns to match the training data
    # Features that were numerical in training, but were passed as categorical in input_df due to get_dummies:
    # These are already numerical and are handled in the initial input_data dict.
    # This reindex step correctly aligns the one-hot encoded columns and ensures numerical columns are in place.
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Drop any columns that are in input_aligned but not in model_columns (shouldn't happen with select boxes)
    extra_columns = set(input_aligned.columns) - set(model_columns)
    if extra_columns:
        st.warning(f"Warning: Unexpected columns found in input and removed: {', '.join(extra_columns)}")
        input_aligned = input_aligned.drop(columns=list(extra_columns))

    # Check for any missing columns that the model expects
    missing_columns = set(model_columns) - set(input_aligned.columns)
    if missing_columns:
        st.error(f"Error: Missing expected input columns: {', '.join(missing_columns)}. This usually indicates a mismatch in column names or options between the app and the trained model.")
        st.stop() # Stop if essential columns are missing

    # Ensure order of columns matches model_columns
    input_aligned = input_aligned[model_columns]

    # Scale the aligned data
    input_scaled = scaler.transform(input_aligned)

    # Make prediction
    prediction = model.predict(input_scaled)

    # Display result
    if prediction[0] == 1:
        st.success("Result: Salary is likely >50K 🎉")
    else:
        st.info("Result: Salary is likely <=50K 😔")

# --- Footer ---
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Scikit-learn.")