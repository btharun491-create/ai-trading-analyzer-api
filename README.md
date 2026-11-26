# AI Trading Analyzer REST API

Flask API for technical indicators, RandomForest direction classification, and stock-news sentiment.

## Important limitations
- This is an analyzer, not a broker or order-execution system.
- It does not guarantee profits; model probabilities are not probabilities of making money.
- The default news sentiment is a lightweight keyword heuristic so the free service has a better chance of fitting in memory. It is not FinBERT.
- FinBERT is optional: install `transformers` and a compatible `torch` build, then set `USE_FINBERT=1` only on a host with sufficient memory.
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
- `GET /api/scan/RELIANCE.NS` - scan one configured stock

## Deploy on Render
Create a Web Service from this repository. Render can use the root `render.yaml`, or set:
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --workers=1 --timeout=180 trading_api.app:app`

The stock scanner trains a model and downloads market data during each scan; requests can take time and may time out if upstream data is slow.
