import pandas as pd
import numpy as np
from .model import prepare_stock, train_and_predict
from .news import get_news_sentiment

STOCKS = {
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "TCS",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "SBIN.NS": "State Bank of India",
    "ITC.NS": "ITC",
    "TMPV.NS": "Tata Motors Passenger Vehicles",
    "BHARTIARTL.NS": "Bharti Airtel",
    "ADANIENT.NS": "Adani Enterprises"
}

CAPITAL = 5000
TOTAL_RISK_PERCENT = 1

def scan_single_stock(symbol):
    company = STOCKS.get(symbol)
    if not company:
        return None

    d = prepare_stock(symbol)
    if d is None or len(d) < 200:
        return None

    up_probability, price, atr = train_and_predict(d)
    if up_probability is None:
        return None

    news = get_news_sentiment(company)
    sentiment = news["sentiment"]
    news_score = news["score"]

    # Conservative Entry filter matching your original notebook
    if (
        up_probability >= 0.65
        and sentiment not in ["NEGATIVE", "UNKNOWN"]
        and news_score > -0.25
    ):
        signal = "BUY"
    else:
        signal = "WAIT"

    stop_distance = atr * 1.5
    risk_budget = CAPITAL * TOTAL_RISK_PERCENT / 100
    
    quantity = int(min(risk_budget / stop_distance, CAPITAL / price)) if stop_distance > 0 else 0
    if signal != "BUY":
        quantity = 0

    return {
        "Stock": symbol,
        "Company": company,
        "Price": round(price, 2),
        "Probability_Up": round(up_probability * 100, 2),
        "News": sentiment,
        "News_Score": round(news_score, 3),
        "Signal": signal,
        "Stop_Loss": round(price - stop_distance, 2),
        "Target": round(price + 2 * stop_distance, 2),
        "Risk_Per_Share": round(stop_distance, 2),
        "Quantity": quantity,
        "Data_Time": str(d.index[-1])
    }
