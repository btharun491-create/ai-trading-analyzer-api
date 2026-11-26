import feedparser
import os
import urllib.parse
from datetime import datetime, timezone, timedelta

# FinBERT pipeline loaded lazily to preserve memory on micro hosting instances
sentiment_pipeline = None

def load_sentiment_model():
    global sentiment_pipeline
    # Lightweight mode is the default for free hosting. Enable FinBERT only
    # on a host with enough RAM by setting USE_FINBERT=1.
    if os.environ.get("USE_FINBERT", "0").strip() != "1":
        return "fallback"
    if sentiment_pipeline is None:
        try:
            from transformers import pipeline
            # Load with cpu/low-memory configuration
            sentiment_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert"
            )
        except Exception as e:
            print(f"Could not load Transformers model: {str(e)}. Falling back to simple heuristic score.")
            sentiment_pipeline = "fallback"
    return sentiment_pipeline

def fallback_sentiment_analysis(titles):
    # Lightweight fallback analyzer in case of memory overflow on micro hosting
    results = []
    positive_words = {'buy', 'bull', 'growth', 'profit', 'rise', 'up', 'win', 'high', 'gains', 'strong', 'positive'}
    negative_words = {'sell', 'bear', 'fall', 'loss', 'down', 'drop', 'risk', 'weak', 'negative', 'crash', 'decline'}
    for title in titles:
        words = set(title.lower().split())
        pos_count = len(words.intersection(positive_words))
        neg_count = len(words.intersection(negative_words))
        if pos_count > neg_count:
            results.append({"label": "positive", "score": 0.65})
        elif neg_count > pos_count:
            results.append({"label": "negative", "score": 0.65})
        else:
            results.append({"label": "neutral", "score": 0.5})
    return results

def get_news_sentiment(company_name):
    query = urllib.parse.quote(company_name + " stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    feed = feedparser.parse(url)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=24)

    articles = []
    for entry in feed.entries:
        published = entry.get("published_parsed")
        if published is None:
            continue

        published_time = datetime(
            published.tm_year, published.tm_mon, published.tm_mday,
            published.tm_hour, published.tm_min, published.tm_sec,
            tzinfo=timezone.utc
        )

        if published_time >= cutoff:
            articles.append({
                "title": entry.title,
                "link": entry.link,
                "published": published_time
            })

    if not articles:
        return {
            "sentiment": "UNKNOWN",
            "score": 0.0,
            "articles": []
        }

    titles = [article["title"] for article in articles]
    
    model = load_sentiment_model()
    if model == "fallback" or model is None:
        results = fallback_sentiment_analysis(titles)
    else:
        try:
            results = model(titles, truncation=True)
        except Exception:
            results = fallback_sentiment_analysis(titles)

    scores = []
    for article, result in zip(articles, results):
        label = result["label"].lower()
        confidence = result["score"]

        if label == "positive":
            score = confidence
        elif label == "negative":
            score = -confidence
        else:
            score = 0.0

        article["sentiment"] = label
        article["score"] = score
        scores.append(score)

    average_score = sum(scores) / len(scores) if scores else 0.0

    if average_score <= -0.25:
        overall = "NEGATIVE"
    elif average_score >= 0.25:
        overall = "POSITIVE"
    else:
        overall = "NEUTRAL"

    # Format timezone datetime back to string representation
    for a in articles:
        a["published"] = a["published"].isoformat()

    return {
        "sentiment": overall,
        "score": average_score,
        "articles": articles
    }
