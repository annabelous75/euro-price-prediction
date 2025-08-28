# Euro price prediction
Interactive Machine Learning project for forecasting EUR/USD exchange rates.
Generates actionable Buy/Sell/Hold signals, visualizes predictions vs actual prices, and demonstrates strong skills in time-series analysis, feature engineering, and data-driven decision making.

# 📈 Project Overview
This project leverages historical EUR/USD exchange rate data to build predictive models using:
1. Linear Regression (LR)
2. Random Forest (RF)
3. Support Vector Regression (SVR)
It provides both short-term (1 hour) and medium-term (48 hours) forecasts, visualizing results in interactive charts and highlighting trading signals for practical decision-making.

# ⚙️ Features
1. Fetches historical EUR/USD data from Yahoo Finance.
2. Computes technical indicators: MA20, MA50, Volatility20, MACD, Signal Line, MACD Histogram, RSI, Percentage Change.
3. Generates predictions for multiple horizons (1h, 48h).
4. Produces actionable trading signals: Buy / Sell / Hold.
5. Interactive visualization with Plotly: compare actual vs predicted prices, highlight forecast points and signals.
6. Saves trained ML models for future use.

# Technologies & Tools
1. Python – Data processing, modeling, automation
2. pandas, NumPy – Data manipulation
3. scikit-learn – ML modeling (Linear Regression, SVR, Random Forest)
4. yfinance – Financial data retrieval
5. Plotly – Interactive charts and dashboards
6. joblib – Model serialization

# 🚀 How to Run Clone the repository:

git clone https://github.com/YourUsername/euro-price-prediction.git

Install dependencies:
pip install -r requirements

Run the main script:
python firstproject.py
