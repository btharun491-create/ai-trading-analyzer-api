import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    "Return", "EMA_9", "EMA_21", "RSI",
    "MACD", "MACD_Signal", "ATR", "Volume_Change"
]

def prepare_stock(symbol):
    try:
        # Download 60 days of 5m data
        raw = yf.download(
            symbol,
            period="60d",
            interval="5m",
            auto_adjust=True,
            progress=False,
            timeout=15
        )

        if raw.empty:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        d = raw.copy()

        d["Return"] = d["Close"].pct_change()
        d["EMA_9"] = d["Close"].ewm(span=9).mean()
        d["EMA_21"] = d["Close"].ewm(span=21).mean()

        delta = d["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        d["RSI"] = 100 - (100 / (1 + rs))

        ema12 = d["Close"].ewm(span=12).mean()
        ema26 = d["Close"].ewm(span=26).mean()

        d["MACD"] = ema12 - ema26
        d["MACD_Signal"] = d["MACD"].ewm(span=9).mean()

        d["ATR"] = (d["High"] - d["Low"]).rolling(14).mean()
        d["Volume_Change"] = d["Volume"].pct_change()

        d["Target"] = (d["Close"].shift(-1) > d["Close"]).astype(int)

        d = d.replace([np.inf, -np.inf], np.nan)
        d = d.dropna()

        return d
    except Exception as e:
        print(f"Error preparing stock data for {symbol}: {str(e)}")
        return None

def train_and_predict(d):
    if d is None or len(d) < 100:
        return None, None, None
        
    # Exclude final row from training labels
    training_data = d.iloc[:-1]
    split = int(len(training_data) * 0.80)

    X_train = training_data[FEATURES].iloc[:split]
    y_train = training_data["Target"].iloc[:split]

    if y_train.nunique() < 2:
        return None, None, None

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    latest = d.iloc[[-1]]
    probability = model.predict_proba(latest[FEATURES])[0]
    up_probability = probability[list(model.classes_).index(1)]

    price = float(latest["Close"].iloc[0])
    atr = float(latest["ATR"].iloc[0])

    return up_probability, price, atr
