import pandas as pd
import joblib
import json
from flask import Flask, request, jsonify, render_template

# Initialize the Flask application
app = Flask(__name__)

# --- LOAD THE SAVED FILES ---
# Load the trained model
model = joblib.load('salary_model.pkl')
# Load the scaler
scaler = joblib.load('scaler.pkl')
# Load the column names
with open('model_columns.json', 'r') as f:
    model_columns = json.load(f)

# --- DEFINE THE WEB PAGES ---
@app.route('/')
def home():
    """Render the home page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receive user input, process it, and return the prediction."""
    # Get the data from the form
    form_data = request.form.to_dict()
    
    # Convert numerical fields from string to number
    # IMPORTANT: Ensure these match the numerical columns in your original dataset
    numerical_features = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    for feature in numerical_features:
        form_data[feature] = float(form_data[feature])
    
    # Create a pandas DataFrame from the user input
    input_df = pd.DataFrame([form_data])
    
    # One-hot encode the input data to match the model's training data
    # This creates columns for all possible categorical values
    input_encoded = pd.get_dummies(input_df)
    
    # Align the columns of the input data with the columns the model was trained on
    # This adds any missing columns (with a value of 0) and ensures the order is the same
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Scale the aligned data using the loaded scaler
    input_scaled = scaler.transform(input_aligned)
    
    # Make a prediction using the loaded model
    prediction = model.predict(input_scaled)
    
    # Determine the prediction text
    prediction_text = 'Salary is likely >50K' if prediction[0] == 1 else 'Salary is likely <=50K'
    
    # Return the result to the HTML page
    return render_template('index.html', prediction_text=prediction_text)

# --- RUN THE APPLICATION ---
if __name__ == '__main__':
    # The app will run on http://127.0.0.1:5000/
    app.run(debug=True)