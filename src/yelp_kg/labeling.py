from __future__ import annotations

import json
import os
from collections import Counter


def heuristic_attribute_label(keywords: list[str]) -> str:
    if not keywords:
        return "general customer experience"

    phrase = ", ".join(keywords[:3])
    mapping = {
        "quiet": "quiet study-friendly space",
        "coffee": "coffee-focused casual hangout",
        "romantic": "romantic date-night atmosphere",
        "bar": "nightlife and bar scene",
        "service": "service quality and staff experience",
        "price": "price and value perception",
        "clean": "cleanliness and comfort",
        "music": "music-driven social atmosphere",
        "wifi": "work-friendly wifi space",
        "parking": "parking and accessibility convenience",
    }
    for token in keywords:
        if token in mapping:
            return mapping[token]
    return f"experience centered on {phrase}"


def maybe_openai_label(
    top_keywords: list[str],
    examples: list[str],
    model_name: str,
) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key)
    prompt = {
        "keywords": top_keywords[:10],
        "examples": examples[:3],
        "task": "Return a short human-readable business attribute label of 2 to 5 words.",
    }
    response = client.responses.create(
        model=model_name,
        input=f"Label this latent Yelp review cluster:\n{json.dumps(prompt, ensure_ascii=True)}",
    )
    text = (response.output_text or "").strip().lower()
    return text[:80] if text else None


def build_attribute_records(
    topic_keywords: dict[int, list[tuple[str, float]]],
    topic_examples: dict[int, list[str]],
    model_name: str | None,
) -> list[dict]:
    attributes: list[dict] = []
    for topic_id, keyword_pairs in sorted(topic_keywords.items()):
        keywords = [keyword for keyword, _ in keyword_pairs[:10]]
        examples = topic_examples.get(topic_id, [])
        label = (
            maybe_openai_label(keywords, examples, model_name)
            if model_name
            else None
        ) or heuristic_attribute_label(keywords)
        attributes.append(
            {
                "attribute_id": f"attr_{topic_id}",
                "topic_id": topic_id,
                "label": label,
                "keywords": keywords,
                "examples": examples,
            }
        )
    return attributes


def summarize_business_attributes(assignments, top_n_topics: int) -> list[dict]:
    topic_counts = (
        assignments[assignments["topic_id"] != -1]
        .groupby(["business_id", "topic_id"])
        .size()
        .reset_index(name="review_count")
    )
    if topic_counts.empty:
        return []

    totals = topic_counts.groupby("business_id")["review_count"].sum().rename("total_topic_reviews").reset_index()
    topic_counts = topic_counts.merge(totals, on="business_id", how="left")
    topic_counts["strength"] = topic_counts["review_count"] / topic_counts["total_topic_reviews"]
    topic_counts = topic_counts.sort_values(["business_id", "strength"], ascending=[True, False])

    records: list[dict] = []
    for business_id, group in topic_counts.groupby("business_id"):
        for row in group.head(top_n_topics).itertuples(index=False):
            records.append(
                {
                    "business_id": business_id,
                    "attribute_id": f"attr_{int(row.topic_id)}",
                    "topic_id": int(row.topic_id),
                    "review_count": int(row.review_count),
                    "strength": float(row.strength),
                }
            )
    return records


def attribute_similarity_edges(attributes: list[dict], limit: int = 15) -> list[dict]:
    records: list[dict] = []
    keyword_sets = {attr["attribute_id"]: set(attr["keywords"]) for attr in attributes}
    ids = [attr["attribute_id"] for attr in attributes]
    for i, left in enumerate(ids):
        for right in ids[i + 1 :]:
            union = keyword_sets[left] | keyword_sets[right]
            if not union:
                continue
            score = len(keyword_sets[left] & keyword_sets[right]) / len(union)
            if score > 0:
                records.append(
                    {
                        "source": left,
                        "target": right,
                        "similarity": round(score, 4),
                    }
                )
    records.sort(key=lambda row: row["similarity"], reverse=True)
    return records[:limit]
