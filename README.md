# Pairs_Trading_Regression_Strategy

## Project Description

This project implements a **statistical arbitrage trading strategy** known as **Pairs Trading** using regression analysis and cointegration tests. The primary goal is to identify stock pairs whose price movements exhibit a stable long-term relationship and exploit temporary deviations from this equilibrium to generate profits.

### Key Highlights:
- **Regression-Based Approach:**  
  Uses Ordinary Least Squares (OLS) regression to model the relationship between two stock prices (X and Y) and calculate the spread (residuals).

- **Cointegration Testing:**  
  Applies the Augmented Dickey-Fuller (ADF) test to verify if the residual spread is stationary — a critical requirement for mean-reverting behavior.

- **Z-Score Signal Generation:**  
  Trading signals are generated when the Z-score of the spread crosses predefined thresholds (entry/exit points).

- **Backtesting Module:**  
  Evaluates historical performance, including metrics like total P&L, win rate, average trade P&L, and cumulative returns over time.

- **Data Fetching:**  
  Historical stock price data is retrieved from **Yahoo Finance (yfinance)** for accurate and reliable backtesting.

- **Customizable Parameters:**  
  The Z-score entry/exit thresholds, lookback windows, and stock pairs are fully configurable for strategy optimization.

This project can serve as a foundation for building **quantitative trading systems**, **algorithmic trading bots**, or as a research tool for **statistical arbitrage strategies**.

---

## Setup Instructions

### 1. Prerequisites
- Python 3.8 or later
- `pip` (Python package manager)
- `git` (to clone the repository)
- A stable internet connection (to fetch live market data from Yahoo Finance)



### 2. Installation Steps

1. **Clone the Repository**
   ```bash
   git clone <https://github.com/Shreyansh100704/Pairs_Trading_Regression_Strategy.git>
   cd <Pairs_Trading_Regression_Strategy>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Mac/Linux
   venv\Scripts\activate  # On Windows
   ```
3. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```


4. Run the app:
   ```bash
   python app.py
   ```
   The application will start on http://127.0.0.1:5000/.
   Open this URL in your browser to access the dashboard.



---

## Models & Technologies Used
- **Backend:** Flask (Python)
- **Data Analysis:** Pandas, NumPy
- **Statistical Modeling:** Statsmodels (OLS Regression, Augmented Dickey-Fuller Test)
- **Market Data:** Yahoo Finance (yfinance)
- **Frontend:** Jinja2 templates (HTML), JavaScript
- **Concurrent Processing:** Python's ProcessPoolExecutor



---

## Strategy Explanation

### **Overview**
The pair trading strategy is based on finding two stocks whose price movements are closely related. When their relative price (spread) deviates beyond a certain threshold, it signals a potential trading opportunity.

---

### **Key Steps**

#### **1. Pair Selection**
Stock pairs are pre-selected based on correlation or sector analysis.

---

#### **2. Cointegration & Regression**
A linear regression (OLS) is performed on one stock's price (Y) against another (X):


Y = alpha + beta * X + epsilon


The residual (epsilon) forms the spread between the two stocks.

---

#### **3. Z-Score Calculation**
The Z-score of the residuals is computed as:


Z = (epsilon - mu)/(sigma)


where mu and sigma are the mean and standard deviation of epsilon.

---

#### **4. Entry & Exit Signals**
- **Entry:** When |Z| > Z_ENTRY (2.5)  
  - If (Z > 2.5), short stock Y and long stock X.
  - If (Z < -2.5), long stock Y and short stock X.

- **Exit:** When |Z| < Z_EXIT (1.0)

---

#### **5. ADF Test for Cointegration**
The Augmented Dickey-Fuller (ADF) test checks if the spread is stationary.  
If **p-value ≥ 0.05**, no trade signal is generated.

---

#### **6. Backtesting**
Historical trade P&L is calculated and visualized to evaluate the performance of each pair.


---
## Files Overview
```
├── app.py                         # Main Flask application with signal generation and analysis logic
├── data/                          # Directory to store user-uploaded data files
│   ├── correlated_pairs.csv       # List of correlated stock pairs
│   ├── final_results.csv          # Backtesting results for all pairs
│   ├── correlated_pairs_by_sector.csv
│   └── correlated_pairs_by_sector_backtest_results.csv
├── templates/                     # HTML templates for the web interface
│   ├── index.html                 # Home page for uploading files and viewing signals
│   └── analysis.html              # Pair-wise P&L and backtest analysis page
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

---
## License
This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this software, provided that proper credit is given to the original authors.
