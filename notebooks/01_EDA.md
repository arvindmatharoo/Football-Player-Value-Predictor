# Exploratory Data Analysis (EDA)
## Football Player Market Value Intelligence System

### 1. Dataset Overview
The dataset contains professional football player statistics collected across multiple leagues and seasons.  
Each row represents a single player with demographic, performance, and contextual attributes used to estimate market value.

**Target Variable:**  
- `market_value` – Estimated monetary value of a player in the transfer market.

---

### 2. Data Dictionary (Key Columns)

| Column Name | Description |
|------------|------------|
| age | Player age (years) |
| position | Primary playing position |
| league | League in which the player competes |
| minutes_played | Total minutes played in the season |
| goals | Total goals scored |
| assists | Total assists |
| appearances | Matches played |
| market_value | Transfer market value (target) |

---

### 3. Missing Value Analysis
- Numerical features show minimal missing values.
- Categorical features (`position`, `league`) are complete.
- Rows with critical missing numerical values were removed to maintain data quality.

---

### 4. Outlier Detection
- Market value shows heavy right skew.
- Extreme high-value players were retained as they represent real-world elite players.
- Outliers were analyzed using boxplots and IQR method.

---

### 5. Distribution Analysis
- **Age:** Concentrated between 20–30 years.
- **Market Value:** Right-skewed distribution.
- **Minutes Played:** Strong indicator of player importance.

---

### 6. Correlation Analysis
Key correlations observed:
- Minutes played ↔ Market value (positive)
- Goals & assists ↔ Market value (positive)
- Age ↔ Market value (non-linear relationship)

---

### 7. Position-wise Analysis
- Forwards and attacking midfielders show higher median market values.
- Defenders and goalkeepers display more stable but lower valuation ranges.

---

### 8. League-wise Analysis
- Players from top-tier leagues show consistently higher market values.
- League strength significantly influences valuation.

---

### 9. Key Insights
- Playing time and attacking contribution strongly influence market value.
- League context is a critical valuation factor.
- Market value is not linearly dependent on age.

---

### 10. Conclusion
EDA confirms that player performance, position, and league context are meaningful predictors of market value and suitable for supervised learning.
