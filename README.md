# Fuel Efficiency Prediction

## Project Overview

This project focuses on developing a machine learning model to predict the fuel consumption of a vehicle based on various features such as mass, engine capacity, fuel type, emissions, energy consumption, and other relevant characteristics. The model helps manufacturers and researchers optimize vehicle designs for better performance.

### Dataset Details

The dataset contains 1 million records from the `Automobile_data` table in the `Database.db` SQLite database:
- **Training Data**: First 700k records.
- **Validation Data**: Next 200k records.
- **Test Data**: Remaining 100k records (used as live data for production evaluation).

### Deliverables

1. Python scripts:
   - `training.py`: For model training and evaluation.
   - `prediction.py`: Flask API for real-time predictions.
   - `testing.py`: For batch evaluation on validation and test data.
2. Final machine learning model saved as `fuel_eff_model.pkl`.
3. PCA transformer for dimensionality reduction saved as `pca_transformer.pkl`.

---

## Features

The following features are included in the dataset:
- `r`: Vehicle type.
- `m (kg)`: Mass of the vehicle in kilograms.
- `Mt`: Transmission type.
- `Ewltp (g/km)`: Emission levels in grams per kilometer.
- `Ft`: Fuel type (e.g., petrol, diesel, electric, LPG).
- `Fm`: Fuel mode.
- `ec (cm3)`: Engine capacity in cubic centimeters.
- `ep (KW)`: Engine power in kilowatts.
- `Erwltp (g/km)`: Real-world emissions in grams per kilometer.
- `Electric range (km)`: Range in kilometers for electric vehicles.

---

## Model Development

### Preprocessing Steps

1. Handling missing values:
   - Features with more than 70% missing values were removed unless significantly correlated with the target variable.
   - Remaining missing values were imputed using median or mode, depending on the data type.
2. Outlier handling using the interquartile range (IQR) method.
3. Encoding categorical features:
   - `Ft` and `Fm` were label-encoded.
4. Feature scaling using `StandardScaler`.
5. Dimensionality reduction using Principal Component Analysis (PCA).

### Model Details

- **Algorithm**: Random Forest Regressor
- **Evaluation Metrics**:
  - Mean Squared Error (MSE)
  - R2 Score

---

## Usage

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Udit11/2025-Fuel-Efficiency-Prediction.git
   cd <repository_name>

2. Install dependencies:
   pip install -r requirements.txt

### Training the Model
Run the training script to preprocess the data, train the model, and save the artifacts:
python training.py
Artifacts generated:
    fuel_eff_model.pkl: Trained model.
    pca_transformer.pkl: PCA transformer.

### Testing the Model
Evaluate the model on validation and test datasets:
python testing.py

### Prediction Using Flask API
1. Start the Flask server:
python prediction.py
2. Send a POST request to the /predict endpoint with input JSON data:
curl -X POST -H "Content-Type: application/json" -d '{"r": ..., "m (kg)": ..., "Mt": ..., "Ewltp (g/km)": ..., "Ft": ..., "Fm": ..., "ec (cm3)": ..., "ep (KW)": ..., "Erwltp (g/km)": ..., "Electric range (km)": ...}' http://127.0.0.1:5000/predict

### Results
Feature Importance: Key predictors include m (kg), ec (cm3), and ep (KW).
Model Performance:
MSE: 0.07310647451241513
R2 Score: 0.8301555674837338
