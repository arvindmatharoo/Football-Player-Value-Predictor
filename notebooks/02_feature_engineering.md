# Feature Engineering
## Football Player Market Value Intelligence System

---

### 1. Feature Selection Rationale
Features were selected based on:
- Domain relevance (football analytics)
- Statistical correlation with market value
- Business interpretability

Selected features include:
- age
- position
- league
- minutes_played
- goals
- assists
- appearances

---

### 2. Handling Categorical Variables

#### Position Encoding
- One-hot encoding applied
- Ensures no ordinal bias between positions

#### League Encoding
- Frequency encoding used
- Represents league strength through player representation

---

### 3. Numerical Feature Processing
- Skewed numerical features analyzed
- Scaling applied where required for linear models
- Tree-based models used raw values

---

### 4. Feature Scaling
- StandardScaler applied for regression-based models
- Tree-based models evaluated without scaling

---

### 5. Train-Test Split Strategy
- 80% training, 20% testing
- Random state fixed for reproducibility
- Stratification considered based on value distribution

---

### 6. Feature Importance (Pre-Model Insight)
Expected high-impact features:
- Minutes played
- Goals & assists
- League representation
- Position (attacking roles)

---

### 7. Final Feature Set
The engineered feature matrix balances:
- Predictive strength
- Interpretability
- Real-world business relevance

---

### 8. Conclusion
Feature engineering ensures the model captures both performance metrics and contextual factors that influence player market valuation.
