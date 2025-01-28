import pandas as pd
import joblib
import sqlite3
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def load_model():
    model = joblib.load('Project 5 (Fuel Efficiency)/fuel_eff_model.pkl')
    pca = joblib.load('Project 5 (Fuel Efficiency)/pca_transformer.pkl')
    return model, pca

# def preprocess_input(data, pca):
#     df = pd.DataFrame(data, index=[0])
#     # df = df[feature_columns]
#     df_scaled = StandardScaler().fit_transform(df)  # Ensure input data is scaled
#     df_pca = pca.transform(df_scaled)  # Apply PCA
#     return df_pca

def load_data():
    conn = sqlite3.connect('Database.db')
    data = pd.read_sql_query("Select * from Automobile_data", conn)
    conn.close()
    return data[700000:900000]

def load_data_test():
    conn = sqlite3.connect('Database.db')
    data = pd.read_sql_query("Select * from Automobile_data", conn)
    conn.close()
    return data[900000:]

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

    X = data.drop('Fuel consumption', axis=1)
    y = data['Fuel consumption']
    
    return X, y

def dim_red(X, pca_final):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_reduced = pca_final.fit_transform(X_scaled)

    return X_reduced

def model_prediction(X, y, pca_final, final_model):
    X_red = dim_red(X, pca_final)
    y_pred = final_model.predict(X_red)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Model Evaluation:\nMSE: {mse}\nR2 Score: {r2}")

def process(data):
    X, y = preprocess_data(data)
    # return model, pca
    final_model, pca_final = load_model()
    model_prediction(X, y, pca_final, final_model)

if __name__ == "__main__":
    data = load_data()
    data_test = load_data_test()
    print("Validation Data\n")
    process(data)
    print("Test Dataset\n")
    process(data_test)