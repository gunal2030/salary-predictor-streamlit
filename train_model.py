# train_model.py

# This script loads the adult 3.csv dataset, preprocesses it,
# trains a RandomForestClassifier and StandardScaler,
# and then saves the trained model, scaler, and feature columns
# to .pkl and .json files.

# Ensure all necessary libraries are installed:
# pip install pandas numpy scikit-learn joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib # Used for saving/loading models and scalers
import json # Used for saving/loading feature column names

print("--- Starting Model Retraining Process ---")

# --- Step 1: Load the dataset ---
# Make sure 'adult 3.csv' is in the same directory as this script.
file_name = 'adult 3.csv'
try:
    df = pd.read_csv(file_name)
    print(f"Dataset '{file_name}' loaded successfully! Rows: {df.shape[0]}, Columns: {df.shape[1]}")
except FileNotFoundError:
    print(f"ERROR: The file '{file_name}' was not found. Please ensure it's in the same folder as train_model.py.")
    exit()

# --- Step 2: Data Cleaning (as per your original code) ---
print("Starting data cleaning...")
df.replace('?', np.nan, inplace=True) # Replace '?' with NaN
for col in ['workclass', 'occupation', 'native-country']:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True) # Fill NaNs with mode

# Convert target variable 'income' to numerical (0 or 1)
df['income'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
print("Data cleaning complete.")

# --- Step 3: Prepare the data for the model (Feature Engineering) ---
print("Preparing data for the model (One-Hot Encoding)...")
X = df.drop('income', axis=1) # Features
y = df['income'] # Target

# Apply one-hot encoding to categorical columns
X = pd.get_dummies(X, drop_first=True)
print(f"Number of feature columns after encoding: {X.shape[1]}")

# --- Step 4: Save the exact order of feature columns ---
# This is CRUCIAL for ensuring consistency when new data comes from the website
# and your app.py processes it.
model_columns = list(X.columns)
with open('model_columns.json', 'w') as f:
    json.dump(model_columns, f)
print("Feature column names saved to 'model_columns.json'.")

# --- Step 5: Split data into training and testing sets ---
# This is standard practice, even if we only need the model trained on full data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split: Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}.")

# --- Step 6: Scale the numerical data ---
print("Scaling numerical features with StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Note: We fit the scaler on X_train, then transform X_train and X_test.
# For deployment, we will save this fitted scaler to use on new, single inputs.
X_test_scaled = scaler.transform(X_test)
print("Scaling complete.")

# --- Step 7: Train the Random Forest Classifier model ---
print("Training the RandomForestClassifier model...")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
print("Model training complete!")

# --- Step 8: Save the trained model and scaler ---
# These files will now be saved with your current scikit-learn version,
# resolving the InconsistentVersionWarning.
joblib.dump(rf_model, 'salary_model.pkl') # Assuming your app.py loads 'salary_model.pkl'
joblib.dump(scaler, 'scaler.pkl')
print("Trained model ('salary_model.pkl') and scaler ('scaler.pkl') saved successfully!")

print("\n--- Model Retraining Process Completed ---")
print("You can now run your Flask application (app.py) without version warnings.")