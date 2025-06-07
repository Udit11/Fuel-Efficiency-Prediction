# 🚗 Fuel Efficiency Prediction

A machine learning solution for predicting vehicle fuel consumption using a variety of technical and environmental features. Designed to support manufacturers and researchers in optimizing automobile designs for performance and sustainability.

---

## 📈 Project Overview

This project focuses on developing a predictive model for **fuel efficiency** using a comprehensive automotive dataset. Key highlights:

- Supervised regression using **Random Forest**
- Preprocessing: missing value handling, encoding, scaling, PCA
- Exposed via a **Flask API** for real-time prediction

---

## 🗃️ Dataset Details

- **Source**: `Automobile_data` table in `Database.db`
- **Records**: 1,000,000 entries
- **Splits**:
  - Training: 700,000 rows
  - Validation: 200,000 rows
  - Testing: 100,000 rows (simulates real-world/live data)

---

## 📦 Deliverables

1. Python scripts:
   - `training.py`: Data preprocessing + model training
   - `testing.py`: Evaluation on validation/test sets
   - `prediction.py`: Flask-based inference API
2. Saved artifacts:
   - `fuel_eff_model.pkl`: Trained model
   - `pca_transformer.pkl`: PCA transformation
3. Dependency list: `requirements.txt`

---

## 🧮 Features Used

| Feature                | Description                                      |
|------------------------|--------------------------------------------------|
| `r`                    | Vehicle type                                     |
| `m (kg)`               | Vehicle mass in kilograms                        |
| `Mt`                   | Transmission type                                |
| `Ewltp (g/km)`         | CO₂ emissions (WLTP standard)                    |
| `Ft`                   | Fuel type (petrol, diesel, electric, LPG, etc.) |
| `Fm`                   | Fuel mode                                        |
| `ec (cm3)`             | Engine capacity                                  |
| `ep (KW)`              | Engine power                                     |
| `Erwltp (g/km)`        | Real-world CO₂ emissions                         |
| `Electric range (km)`  | EV range (if applicable)                         |

---

## 🧪 Model Development

### 🔧 Preprocessing Steps

- **Missing Values**:
  - Drop features with >70% missing unless highly correlated with target
  - Median/Mode imputation for remaining
- **Outliers**:
  - Removed using the IQR method
- **Encoding**:
  - Label Encoding for categorical (`Ft`, `Fm`)
- **Scaling**:
  - `StandardScaler`
- **Dimensionality Reduction**:
  - PCA for noise reduction and performance boost

### 🤖 Model Configuration

- **Model**: Random Forest Regressor
- **Metrics**:
  - Mean Squared Error (MSE)
  - R² Score

---

## 🛠️ Usage Guide

### 🔧 Installation

```bash
git clone https://github.com/Udit11/Fuel-Efficiency-Prediction.git
cd Fuel-Efficiency-Prediction
pip install -r requirements.txt
```

---

### 🏋️‍♂️ Train the Model

```bash
python training.py
```

**Artifacts Saved:**
- `fuel_eff_model.pkl`
- `pca_transformer.pkl`

---

### 📊 Evaluate the Model

```bash
python testing.py
```

Used for both validation and test set evaluation.

---

### 🌐 Prediction via Flask API

1. **Start the Server:**

```bash
python prediction.py
```

2. **Send a POST Request:**

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "r": "...",
  "m (kg)": 1350,
  "Mt": "...",
  "Ewltp (g/km)": 105.3,
  "Ft": "petrol",
  "Fm": "...",
  "ec (cm3)": 1498,
  "ep (KW)": 88,
  "Erwltp (g/km)": 112.5,
  "Electric range (km)": 0
}' http://127.0.0.1:5000/predict
```

---

## 📊 Results

- **Key Predictors**: `m (kg)`, `ec (cm3)`, `ep (KW)`
- **Performance**:
  - **MSE**: 0.0731
  - **R² Score**: 0.8302

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Udit Srivastava**  
AI/ML Engineer | MSc in Computing (AI), Dublin City University  
📧 Email: uditsrivastava2347@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/udit-srivastava/)

---
