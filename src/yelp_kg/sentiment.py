from __future__ import annotations

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def score_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    scored = reviews.copy()
    compounds: list[float] = []
    labels: list[str] = []

    for text in scored["text"].astype(str):
        compound = float(analyzer.polarity_scores(text)["compound"])
        compounds.append(compound)
        if compound >= 0.05:
            labels.append("positive")
        elif compound <= -0.05:
            labels.append("negative")
        else:
            labels.append("neutral")

    scored["sentiment_score"] = compounds
    scored["sentiment_label"] = labels
    return scored


def summarize_attribute_sentiment(assignments: pd.DataFrame) -> list[dict]:
    valid = assignments[assignments["topic_id"] != -1].copy()
    if valid.empty:
        return []

    grouped = (
        valid.groupby("topic_id")
        .agg(
            review_count=("review_id", "count"),
            avg_sentiment=("sentiment_score", "mean"),
            positive_share=("sentiment_label", lambda s: (s == "positive").mean()),
            neutral_share=("sentiment_label", lambda s: (s == "neutral").mean()),
            negative_share=("sentiment_label", lambda s: (s == "negative").mean()),
        )
        .reset_index()
    )

    records: list[dict] = []
    for row in grouped.itertuples(index=False):
        records.append(
            {
                "attribute_id": f"attr_{int(row.topic_id)}",
                "topic_id": int(row.topic_id),
                "review_count": int(row.review_count),
                "avg_sentiment": round(float(row.avg_sentiment), 4),
                "positive_share": round(float(row.positive_share), 4),
                "neutral_share": round(float(row.neutral_share), 4),
                "negative_share": round(float(row.negative_share), 4),
            }
        )
    return records


def summarize_business_sentiment(assignments: pd.DataFrame) -> list[dict]:
    if assignments.empty:
        return []

    grouped = (
        assignments.groupby("business_id")
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            positive_share=("sentiment_label", lambda s: (s == "positive").mean()),
            neutral_share=("sentiment_label", lambda s: (s == "neutral").mean()),
            negative_share=("sentiment_label", lambda s: (s == "negative").mean()),
        )
        .reset_index()
    )

    records: list[dict] = []
    for row in grouped.itertuples(index=False):
        records.append(
            {
                "business_id": row.business_id,
                "avg_sentiment": round(float(row.avg_sentiment), 4),
                "positive_share": round(float(row.positive_share), 4),
                "neutral_share": round(float(row.neutral_share), 4),
                "negative_share": round(float(row.negative_share), 4),
            }
        )
    return records
