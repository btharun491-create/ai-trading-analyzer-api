import os
from flask import Flask, jsonify
from flask_cors import CORS
from .scanner import STOCKS, scan_single_stock

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        "name": "AI Trading Analyzer REST API",
        "description": "Independent cloud API evaluating 10 top NSE stocks using RandomForest ML models & FinBERT News Sentiment Analysis.",
        "endpoints": {
            "/health": "Check service health status.",
            "/api/scan": "Run multi-stock scanner for all 10 NSE stocks.",
            "/api/scan/<stock_symbol>": "Detailed analytics for a specific stock (e.g. /api/scan/RELIANCE.NS)"
        },
        "disclaimer": "Prediction model is not guaranteed. Verify real-time metrics before making manual trading decisions."
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "uptime": "active"})

@app.route('/api/scan', methods=['GET'])
def api_scan_all():
    results = []
    for symbol in STOCKS.keys():
        res = scan_single_stock(symbol)
        if res:
            results.append(res)
    return jsonify(results)

@app.route('/api/scan/<symbol>', methods=['GET'])
def api_scan_one(symbol):
    symbol_upper = symbol.upper()
    if symbol_upper not in STOCKS:
        return jsonify({"error": f"Symbol '{symbol}' is not configured in the 10-stock watch list."}), 404
    
    res = scan_single_stock(symbol_upper)
    if not res:
        return jsonify({"error": f"Failed to fetch enough market data or calculate indicators for {symbol_upper}."}), 500
        
    return jsonify(res)

if __name__ == '__main__':
    # Bind to PORT environment variable for Render cloud deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
