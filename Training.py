import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data():
    conn = sqlite3.connect('Database.db')
    data = pd.read_sql_query("Select * from Automobile_data", conn)
    conn.close()
    return data

def preprocess_data(data):
    data.rename(columns={'Fuel consumption ': 'Fuel consumption'}, inplace=True)

    missing_percentage = data.isnull().mean() * 100
    high_missing_cols = missing_percentage[missing_percentage > 70].index.tolist()

    target = 'Fuel consumption'

    numerical_cols = data.select_dtypes(include=[np.number]).columns
    high_missing_numerical_cols = [col for col in high_missing_cols if col in numerical_cols]

    correlations = data[high_missing_numerical_cols + [target]].corr()[target]

    significant_threshold = 0.3
    significant_cols = correlations[abs(correlations) > significant_threshold].index.tolist()
    columns_to_drop = [col for col in high_missing_cols if col not in significant_cols]
    data_cleaned = data.drop(columns=columns_to_drop)

    data['m (kg)'].fillna(data['m (kg)'].median(), inplace=True)
    data['Mt'].fillna(data['Mt'].mean(), inplace=True)
    data['Ewltp (g/km)'].fillna(data['Ewltp (g/km)'].median(), inplace=True)
    data['Ft'].fillna(data['Ft'].mode()[0], inplace=True)
    data['Fm'].fillna(data['Fm'].mode()[0], inplace=True)
    data['ec (cm3)'].fillna(data['ec (cm3)'].median(), inplace=True)
    data['ep (KW)'].fillna(data['ep (KW)'].median(), inplace=True)
    data.drop(columns=['z (Wh/km)'], inplace=True)
    data['Erwltp (g/km)'].fillna(data['Erwltp (g/km)'].median(), inplace=True)
    data['Fuel consumption'].fillna(data['Fuel consumption'].mean(), inplace=True)
    data['Electric range (km)'].fillna(data['Electric range (km)'].median(), inplace=True)

    outlier_columns = ['m (kg)', 'Mt', 'Ewltp (g/km)', 'ec (cm3)', 'ep (KW)', 'Erwltp (g/km)', 'Fuel consumption', 'Electric range (km)']

    for col in outlier_columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Capping/Flooring outliers
        data[col] = data[col].clip(lower_bound, upper_bound)
    
    le = LabelEncoder()

    # Apply to Ft and Fm
    data['Ft'] = le.fit_transform(data['Ft'])
    data['Fm'] = le.fit_transform(data['Fm'])

    train_data = data[:700000]

    X = train_data.drop('Fuel consumption', axis=1)
    y = train_data['Fuel consumption']
    
    return X, y

def dim_red(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_final = PCA(n_components=6)
    X_reduced = pca_final.fit_transform(X_scaled)

    return X_reduced, pca_final

def train_model(X, y, original_columns = None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features='sqrt',
        max_depth=20,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate Model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Model Evaluation:\nMSE: {mse}\nR2 Score: {r2}")

    if original_columns is not None:
        return model, original_columns
    else:
        # If original columns are not provided, return indices
        return model, range(X.shape[1])

# def test_model(X, y):
#     y_pred_pca = model.predict(X)
#     print(f"Mean Squared Error on PCA-transformed validation data: {mean_squared_error(y, y_pred_pca)}")
#     print(f"R^2 Score on PCA-transformed validation data: {r2_score(y, y_pred_pca)}")

def save_model(model, feature_columns, pca):
    joblib.dump(model, 'Project 5 (Fuel Efficiency)/fuel_eff_model.pkl')
    joblib.dump(feature_columns, 'Project 5 (Fuel Efficiency)/feature_columns.pkl')
    joblib.dump(pca, 'Project 5 (Fuel Efficiency)/pca_transformer.pkl')

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data()
    X, y = preprocess_data(data)
    X_PCA, final_PCA = dim_red(X)

    # Train and save model
    model, feature_columns = train_model(X_PCA, y)
    save_model(model, feature_columns, final_PCA)

    print("Model and feature columns saved successfully")