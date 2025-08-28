import yfinance as yf 
import pandas as pd
import os
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import numpy as np
import plotly.graph_objects as go

# ===== Функции =====
def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def prepare_data(ticker: str = 'EURUSD=X', period: str = '6mo', interval: str = '1h', horizon: int = 48):
    data = yf.download(ticker, period=period, interval=interval)
    if data.empty:
        raise ValueError(f"No data found for {ticker}")
    data.columns = data.columns.get_level_values(0)

    # Целевая переменная
    data[f'Close_t+{horizon}'] = data['Close'].shift(-horizon)

    # Технические индикаторы
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['Volatility20'] = data['Close'].rolling(window=20).std()
    data['MA_diff'] = data['MA20'] - data['MA50']
    data['Pct_change'] = data['Close'].pct_change()

    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    data['MACD'] = macd
    data['Signal_line'] = signal
    data['MACD_histogram'] = macd - signal
    data['RSI'] = calculate_rsi(data['Close'])

    features = ['Close', 'MA20', 'MA50', 'RSI', 'Volatility20', 'MA_diff', 'Pct_change', 'Volume', 
                'MACD', 'Signal_line', 'MACD_histogram', f'Close_t+{horizon}']

    data_clean = data.dropna(subset=features)
    return data_clean, features

def train_and_save_models(X_train, y_train, save_dir="models"):
    os.makedirs(save_dir, exist_ok=True)

    models = {
        "LinearRegression": LinearRegression(),
        "SVR": SVR(),
    }

    # 🔹 GridSearch + TimeSeriesSplit for RandomForest
    rf_param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10]
    }

    tscv = TimeSeriesSplit(n_splits=5)

    rf_grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=42),
        param_grid=rf_param_grid,
        cv=tscv,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    models["RandomForest"] = rf_grid

    # 🔹 Train and save
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        if isinstance(model, GridSearchCV):
            print(f"Best params for {name}: {model.best_params_}")
            best_model = model.best_estimator_
        else:
            best_model = model

        joblib.dump(best_model, os.path.join(save_dir, f"{name}.pkl"))
        print(f"{name} saved to {save_dir}/{name}.pkl")

    return models

def predict_and_visualize(models: dict, X_test: pd.DataFrame, y_test: pd.Series, horizon: int = 48, threshold: float = 0.001):
    lr_model = models['LinearRegression']
    rf_model = models['RandomForest']

    y_pred_lr = lr_model.predict(X_test)

    # Сигналы
    signals = ['Buy' if diff > threshold else 'Sell' if diff < -threshold else 'Hold' 
               for diff in (y_pred_lr - y_test.values)]
    signals_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_lr, 'Signal': signals}, index=y_test.index)

    # Визуализация
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Actual Close Price'))
    fig.add_trace(go.Scatter(x=y_test.index, y=y_pred_lr, mode='lines', name='Predicted Close Price (LR)', line=dict(dash='dash')))

    buy_signals = signals_df[signals_df['Signal'] == 'Buy']
    sell_signals = signals_df[signals_df['Signal'] == 'Sell']

    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Actual'], mode='markers', 
                             marker=dict(symbol='triangle-up', color='green', size=10), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Actual'], mode='markers', 
                             marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal'))

    last_date = X_test.index[-1]
    forecast_price = y_pred_lr[-1]
    fig.add_trace(go.Scatter(x=[last_date + pd.Timedelta(hours=horizon)], y=[forecast_price], 
                             mode='markers+text', marker=dict(symbol='circle', color='blue', size=15), 
                             text=[f'{forecast_price:.4f}'], textposition='top center', name=f'Forecast in {horizon} steps'))

    fig.update_layout(title=f'EUR/USD Forecast (horizon={horizon})', xaxis_title='Date', yaxis_title='Price', template='plotly_white')
    fig.show()

    print(f'MSE: {mean_squared_error(y_test, y_pred_lr):.6f}')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
    print(f'RMSE: {rmse:.6f}')
    print(f'MAE: {mean_absolute_error(y_test, y_pred_lr):.6f}')
    print(f'R2: {r2_score(y_test, y_pred_lr):.6f}')

# ===== Главный блок =====
if __name__ == '__main__':
    threshold = 0.001

    # ===== Прогноз на 1 час =====
    data_1h, features_1h = prepare_data(horizon=1)
    X_1h = data_1h[features_1h[:-1]]
    y_1h = data_1h[features_1h[-1]]
    models_1h = train_and_save_models(X_1h, y_1h)
    print("Прогноз на 1 час:")
    predict_and_visualize(models_1h, X_1h, y_1h, horizon=1, threshold=threshold)

    # ===== Прогноз на 2 дня (48 часов) =====
    data_48h, features_48h = prepare_data(horizon=48)
    X_48h = data_48h[features_48h[:-1]]
    y_48h = data_48h[features_48h[-1]]
    models_48h = train_and_save_models(X_48h, y_48h)
    print("\nПрогноз на 2 дня:")
    predict_and_visualize(models_48h, X_48h, y_48h, horizon=48, threshold=threshold)
