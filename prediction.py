import pandas as pd
import joblib
from flask import Flask, request, jsonify
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)

# Preprocessing function (adapted from your provided code)
def preprocess_input(data, feature_columns, pca):
    # Convert input JSON to DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Ensure the input data has only the required features
    df = df[feature_columns]
    
    # Handle any preprocessing steps like encoding or scaling
    df['Ft'] = df['Ft'].map({'petrol': 0, 'diesel': 1, 'electric': 2, 'lpg': 3})  # Example encoding
    df['Fm'] = df['Fm'].map({'M': 0, 'E': 1, 'B': 2})  # Example encoding
    
    # Apply scaling
    df_scaled = StandardScaler().fit_transform(df)
    
    # Apply PCA
    df_pca = pca.transform(df_scaled)
    
    return df_pca

# Load model and PCA
def load_model():
    model = joblib.load('Project 5 (Fuel Efficiency)/fuel_eff_model.pkl')
    pca = joblib.load('Project 5 (Fuel Efficiency)/pca_transformer.pkl')
    # feature_columns = joblib.load('Project 5 (Fuel Efficiency)/feature_columns.pkl')
    return model, pca

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        model, pca = load_model()

        feature_columns = ['r', 'm (kg)', 'Mt', 'Ewltp (g/km)', 'Ft', 'Fm', 'ec (cm3)', 'ep (KW)', 'Erwltp (g/km)', 'Electric range (km)']
        
        # Preprocess input data
        processed_data = preprocess_input(input_data, feature_columns, pca)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        return jsonify({"predicted_fuel_consumption": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
