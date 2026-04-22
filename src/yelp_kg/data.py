from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd

from .config import DatasetPaths, PipelineConfig


def stream_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_businesses(config: PipelineConfig) -> pd.DataFrame:
    rows: list[dict] = []
    for row in stream_jsonl(config.paths.business_path):
        state = row.get("state") or ""
        categories = row.get("categories") or ""
        if config.state_filter and state.lower() != config.state_filter.lower():
            continue
        if config.city_filter and row.get("city", "").lower() != config.city_filter.lower():
            continue
        if config.category_filter and config.category_filter.lower() not in categories.lower():
            continue
        rows.append(
            {
                "business_id": row["business_id"],
                "name": row.get("name"),
                "city": row.get("city"),
                "state": state,
                "stars": row.get("stars"),
                "review_count": row.get("review_count", 0),
                "categories": categories,
                "is_open": row.get("is_open"),
            }
        )
    businesses = pd.DataFrame(rows)
    if businesses.empty:
        return businesses
    return businesses[businesses["review_count"] >= config.min_business_reviews].reset_index(drop=True)


def reservoir_sample_reviews(config: PipelineConfig, allowed_business_ids: set[str]) -> pd.DataFrame:
    rng = random.Random(config.random_seed)
    sample: list[dict] = []
    all_reviews: list[dict] = []
    seen = 0
    for row in stream_jsonl(config.paths.review_path):
        business_id = row.get("business_id")
        text = (row.get("text") or "").strip()
        if business_id not in allowed_business_ids:
            continue
        if len(text) < config.min_review_length:
            continue

        entry = {
            "review_id": row["review_id"],
            "business_id": business_id,
            "user_id": row.get("user_id"),
            "stars": row.get("stars"),
            "date": row.get("date"),
            "text": text.replace("\r", " ").replace("\n", " "),
            "useful": row.get("useful", 0),
            "funny": row.get("funny", 0),
            "cool": row.get("cool", 0),
        }

        if config.use_all_reviews:
            all_reviews.append(entry)
            continue

        seen += 1
        if len(sample) < config.sample_size:
            sample.append(entry)
            continue

        idx = rng.randint(0, seen - 1)
        if idx < config.sample_size:
            sample[idx] = entry

    reviews = pd.DataFrame(all_reviews if config.use_all_reviews else sample)
    if not reviews.empty:
        reviews = reviews.drop_duplicates(subset=["review_id"]).reset_index(drop=True)
    return reviews


def load_tips(paths: DatasetPaths, allowed_business_ids: set[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for row in stream_jsonl(paths.tip_path):
        if row.get("business_id") not in allowed_business_ids:
            continue
        rows.append(
            {
                "business_id": row["business_id"],
                "likes": row.get("likes", 0),
                "tip_text": (row.get("text") or "").strip(),
            }
        )
    return pd.DataFrame(rows)


def load_checkins(paths: DatasetPaths, allowed_business_ids: set[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for row in stream_jsonl(paths.checkin_path):
        business_id = row.get("business_id")
        if business_id not in allowed_business_ids:
            continue
        date_text = row.get("date") or ""
        rows.append(
            {
                "business_id": business_id,
                "checkin_count": len([part for part in date_text.split(",") if part.strip()]),
            }
        )
    return pd.DataFrame(rows)


def aggregate_business_statistics(
    businesses: pd.DataFrame,
    reviews: pd.DataFrame,
    tips: pd.DataFrame,
    checkins: pd.DataFrame,
) -> pd.DataFrame:
    base = businesses.copy()

    review_stats = (
        reviews.groupby("business_id")
        .agg(
            sampled_review_count=("review_id", "count"),
            sampled_avg_review_stars=("stars", "mean"),
            sampled_total_useful=("useful", "sum"),
        )
        .reset_index()
    )
    tip_stats = (
        tips.groupby("business_id")
        .agg(
            tip_count=("tip_text", "count"),
            tip_likes=("likes", "sum"),
        )
        .reset_index()
        if not tips.empty
        else pd.DataFrame(columns=["business_id", "tip_count", "tip_likes"])
    )

    merged = base.merge(review_stats, on="business_id", how="left")
    merged = merged.merge(tip_stats, on="business_id", how="left")
    merged = merged.merge(checkins, on="business_id", how="left")

    for col in ["sampled_review_count", "sampled_avg_review_stars", "sampled_total_useful", "tip_count", "tip_likes", "checkin_count"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    return merged


def top_keywords_per_business(reviews: pd.DataFrame, limit: int = 8) -> dict[str, list[str]]:
    stop_words = {
        "the", "and", "was", "were", "with", "this", "that", "have", "from", "they",
        "very", "just", "been", "into", "about", "there", "would", "could", "their",
        "place", "really", "because", "when", "what", "where", "which", "while", "after",
    }
    tokens_by_business: dict[str, Counter[str]] = defaultdict(Counter)
    for row in reviews.itertuples(index=False):
        for token in str(row.text).lower().split():
            token = token.strip(".,!?;:\"'()[]{}")
            if len(token) < 4 or token in stop_words or not token.isascii():
                continue
            tokens_by_business[row.business_id][token] += 1
    return {
        business_id: [token for token, _ in counts.most_common(limit)]
        for business_id, counts in tokens_by_business.items()
    }
