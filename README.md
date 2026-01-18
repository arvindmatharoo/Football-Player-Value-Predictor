# ⚽ Football Market Value Intelligence System

An end-to-end **Data Science & Machine Learning project** that predicts **football player market value tiers (Low / Mid / High)** using player performance, availability, age, and contextual information — built from raw data to a **production-ready FastAPI deployment**.

---

## 📌 Problem Statement

Football player market value is influenced by more than just goals.  
Availability, consistency, age, role, and playing context all contribute to valuation.

This project models **relative market value tiers** using data-driven insights when direct market value data is unavailable.

---

## 🎯 Objectives

- Design a **proxy target variable** for market value
- Perform deep **EDA and feature engineering**
- Train interpretable ML models
- Validate assumptions using **ablation study**
- Deploy a **production-ready ML API**

---

## 🗂️ Dataset

- **Source**: Kaggle – Top 5 European Leagues Player Stats (2022–23)
- **Players**: 2,769 → filtered to **2,029**
- **Leagues**:
  - Premier League  
  - La Liga  
  - Serie A  
  - Bundesliga  
  - Ligue 1  

### Key Raw Features
- Age, minutes played, starts
- Goals, assists, xG, xAG (per 90)
- Progressive carries & passes
- Position and league

---

## 🔍 Exploratory Data Analysis (EDA)

Performed:
- Univariate, bivariate & multivariate analysis
- Correlation analysis
- Distribution and skewness checks

### Key Insights
- Performance metrics are heavily skewed
- Minutes played is a major confounder
- Goals alone are poor indicators of market value
- Mid-tier players are hardest to classify

---

## 🧠 Target Variable Design

Since real market value is unavailable, a **proxy target** was engineered.

### Value Score Components
1. **Performance Efficiency**  
   - Expected Goals (xG)  
   - Expected Assisted Goals (xAG)

2. **Availability / Trust**  
   - Minutes played  
   - Starts

3. **Market Logic**  
   - Age (younger players valued higher for same output)

### Final Target
- `value_tier` ∈ **{Low, Mid, High}**
- Created using **quantile-based binning**
- Balanced class distribution

---

## ⚙️ Feature Engineering

Raw features were intentionally **compressed into higher-level signals** to reduce noise and multicollinearity.

### Final Features
- `age_years`
- `min`
- `availability_ratio`
- `offensive_efficiency`
- `progressive_actions`
- `position`
- `compition`

---

## 🤖 Modeling

### Models Used
- Logistic Regression (baseline)
- Random Forest Classifier

### Performance
| Model | Accuracy | Macro F1 |
|------|----------|----------|
| Logistic Regression | ~0.90 | ~0.90 |
| Random Forest | ~0.90 | ~0.90 |

The similar performance indicates strong feature quality and near-linear separability.

---

## 🔬 Ablation Study

To understand feature importance, features were removed systematically.

| Scenario | Accuracy |
|--------|----------|
| All Features | **0.904** |
| No Availability | **0.685** |
| No Performance | **0.749** |
| No Age | **0.764** |
| No Context (Position + League) | **0.889** |

### Key Findings
- Availability is the strongest driver of market value
- Performance matters when supported by playing time
- Age has an independent market effect
- Context fine-tunes valuation

---

## 🚀 Deployment (FastAPI)

A production-ready **REST API** was built using **FastAPI**.

### Sample Request 
```json
{
"age_years": 24,
  "min": 2000,
  "availability_ratio": 0.8,
  "offensive_efficiency": 0.35,
  "progressive_actions": 4.5,
  "position": "MF",
  "compition": "Premier League"
}
```
### Sample Response
```json
{
"predicted_value_tier" : "Mid"
}
```

## Tech Stack 
- Python
- Pandas, Numpy
- Scikit-Learn
- FastAPI
- Uvicorn
- Joblib
- Matplotlib,Seaborn

## Future Improvements
- Team-strength proxy features
- Probability-based prediction
- Model monitoring & drift detection
- Dockerization and Cloud deployment
- Frontend Dashboard

## Author 
- **Arvind Singh**
- **B. Tech Computer Science Engineering**
- [Linkedin](https://www.linkedin.com/in/arvindmatharoo/)
- Gmail: iarvinddsingh@gmail.com


