# AI Trading Analyzer REST API

Flask API for technical indicators, RandomForest direction classification, and stock-news sentiment.

## Important limitations
- Yahoo Finance data may be delayed, incomplete, or unavailable. This API is not guaranteed to provide real-time market data.
- Free hosting can sleep and may have cold-start delays.

## Run locally
From the project root:
```bash
python -m venv .venv
# activate the environment, then:
pip install -r requirements.txt
python -m trading_api.app
```

## Endpoints
- `GET /` - API information
- `GET /health` - health check
- `GET /api/scan` - scan configured stocks
- `GET /api/scan/<stock_symbol>` - scan one configured stock (e.g., `GET /api/scan/RELIANCE.NS`)

## Deploy on Render
**Live Deployment URL:** https://ai-trading-analyzer-api.onrender.com

Create a Web Service from this repository. Render can use the root `render.yaml`, or set:
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --workers=1 --timeout=180 trading_api.app:app`

The stock scanner trains a model and downloads market data during each scan; requests can take time and may time out if upstream data is slow.

## Sample JSON Response
```json
{
  "status": "success",
  "symbol": "RELIANCE.NS",
  "signal": "BUY",
  "confidence": 0.85,
  "metrics": {
    "rsi": 42.5,
    "sma_50": 2450.0
  }
}
```
