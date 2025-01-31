import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model('gru_pm10_model.h5')
scaler = joblib.load('scaler.pkl')

# Define a function to preprocess input data
def preprocess_input(input_data):
    # Convert the input data to DataFrame
    input_df = pd.DataFrame(input_data)

    # Drop non-numeric columns (excluding datetime)
    non_numeric_columns = [col for col in input_df.columns if input_df[col].dtype == 'object' or not np.issubdtype(input_df[col].dtype, np.number)]
    input_df.drop(columns=non_numeric_columns, inplace=True)
    
    # Fill missing values
    input_df.fillna(input_df.mean(), inplace=True)

    # Scale the data
    input_scaled = scaler.transform(input_df)
    
    return input_scaled, input_df

# Define function to create sequences
def create_sequences(data, time_step):
    X = []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step, :-1])  # All columns except the last (target)
    return np.array(X)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        input_data = request.get_json()
        
        # Preprocess the input data and get the original DataFrame
        input_scaled, input_df = preprocess_input(input_data)
        
        # Create sequences for prediction
        time_step = 36  # Using time_step as in your model
        X_input = create_sequences(input_scaled, time_step)

        # Make predictions
        predictions = model.predict(X_input)
        
        # Rescale predictions to original scale
        predictions_rescaled = scaler.inverse_transform(np.concatenate([np.zeros((predictions.shape[0], input_df.shape[1] - 1)), predictions], axis=1))[:, -1]
        
        # Return predictions as JSON
        return jsonify({'predictions': predictions_rescaled.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
