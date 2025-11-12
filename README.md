# ⚽ Player Market Value Predictor

> A machine learning–powered project that predicts the **market value of football players (€)** using FIFA-style player attributes — built with **Python, Scikit-learn, and Streamlit**.

---

## 📘 Overview

This project applies **Machine Learning** to estimate a football player's market value based on various attributes such as skill ratings, physical characteristics, and playstyle.

The model uses a **Random Forest Regression** pipeline that automatically handles scaling and encoding.  
A modern **Streamlit web app** provides an interactive interface for both single-player and batch predictions.

---

## 🧠 Key Features

- ✅ Predicts player market value in euros (€)  
- ✅ Single-player input or batch CSV upload  
- ✅ Displays model confidence & prediction spread  
- ✅ Visualizes predicted value distribution  
- ✅ Built-in preprocessing (scaling, encoding, imputation)  
- ✅ Ready for deployment via Streamlit

---

## 🗂️ Project Structure

📦 football-value-predictor/
├── app.py # Streamlit web app
├── value_predictor_pipeline.pkl # Trained ML pipeline (model + preprocessing)
├── fifa_players.csv # Dataset used for training (optional)
├── README.md # Project documentation
├── requirements.txt # Dependencies
└── notebook.ipynb # Model training notebook


---

## 🧩 Model Workflow

1. **Data Cleaning**
   - Removed missing and irrelevant fields.
   - Filled numerical missing values using median imputation.
   - Dropped columns not used for modeling.

2. **Feature Engineering**
   - Selected 39 total features (35 numeric, 4 categorical).
   - Applied `log1p()` transformation to target (`value_euro`) to normalize skew.

3. **Preprocessing**
   - Numeric: `StandardScaler`  
   - Categorical: `OneHotEncoder(handle_unknown='ignore')`

4. **Model Training**
   - Algorithm: `RandomForestRegressor (n_estimators=200)`
   - Evaluation (on test set):
     - **MAE:** €139,486  
     - **RMSE:** €960,433  
     - **R²:** 0.9722  

5. **Deployment**
   - Model saved using `joblib` as a single `.pkl` pipeline.
   - Integrated into an interactive **Streamlit** app.

---

## 🚀 How to Run Locally

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/football-value-predictor.git
cd football-value-predictor
```
### Create and activate a Virtual Environment (optional)
```bash
conda create -n football-predictor python=3.10
conda activate football-predictor
```
or using venv:
```bash
python -m venv venv
source venv/bin/activate
```

### Install dependencies 
```bash
pip install "numpy<2.0" pandas scikit-learn joblib streamlit plotly
```
### Run the streamlit app 
```bash
streamlit run app.py
```

# Usage
1. **Single Player Mode**
   - Fill in player details (age, height, skills, etc.)
   - Choose options for categorical features (foot, position, nationality)
   - Click Predict Player Value
   - he app shows predicted value, confidence, and approximate standard deviation
2. **Batch Mode**
   - Upload a CSV file containing the same features used during training
   - The app displays predictions in a table
   - Download results as a CSV file and view prediction distribution

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| **Language** | Python 3.10 |
| **Machine Learning Framework** | Scikit-learn |
| **Web Framework (UI)** | Streamlit |
| **Data Handling** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib |
| **Model Serialization** | Joblib |
| **Environment Management** | Conda / Virtualenv |

