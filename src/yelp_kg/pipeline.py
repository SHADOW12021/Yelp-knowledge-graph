from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .data import (
    aggregate_business_statistics,
    load_businesses,
    load_checkins,
    load_tips,
    reservoir_sample_reviews,
    top_keywords_per_business,
)
from .graph_builder import build_graph, export_graph
from .labeling import attribute_similarity_edges, build_attribute_records, summarize_business_attributes
from .query import build_query_index
from .sentiment import score_reviews, summarize_attribute_sentiment, summarize_business_sentiment
from .topic_modeling import discover_topics


def _write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _format_elapsed(seconds: float) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours >= 1:
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"
    if minutes >= 1:
        return f"{int(minutes)}m {secs:.1f}s"
    return f"{secs:.1f}s"


def _log_stage(stage_name: str, started_at: float) -> None:
    print(f"[pipeline] {stage_name} done in {_format_elapsed(time.perf_counter() - started_at)}.")


def run_pipeline(config: PipelineConfig) -> dict:
    total_started_at = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pipeline] Starting run. Output dir: {config.output_dir}")

    print("[pipeline] Loading businesses...")
    stage_started_at = time.perf_counter()
    businesses = load_businesses(config)
    if businesses.empty:
        raise ValueError("No businesses matched the current filters.")
    print(f"[pipeline] Loaded {len(businesses):,} businesses after filters.")
    _log_stage("Loading businesses", stage_started_at)

    allowed_business_ids = set(businesses["business_id"])
    print("[pipeline] Loading reviews...")
    stage_started_at = time.perf_counter()
    reviews = reservoir_sample_reviews(config, allowed_business_ids)
    if reviews.empty:
        raise ValueError("No reviews were sampled. Relax the filters or lower review thresholds.")
    print(f"[pipeline] Loaded {len(reviews):,} reviews for modeling.")
    _log_stage("Loading reviews", stage_started_at)

    print("[pipeline] Scoring review sentiment...")
    stage_started_at = time.perf_counter()
    reviews = score_reviews(reviews)
    _log_stage("Scoring review sentiment", stage_started_at)

    print("[pipeline] Loading tips and check-ins...")
    stage_started_at = time.perf_counter()
    tips = load_tips(config.paths, allowed_business_ids)
    checkins = load_checkins(config.paths, allowed_business_ids)
    print(f"[pipeline] Loaded {len(tips):,} tips and {len(checkins):,} check-in rows.")
    _log_stage("Loading tips and check-ins", stage_started_at)

    print("[pipeline] Aggregating business statistics and keywords...")
    stage_started_at = time.perf_counter()
    business_stats = aggregate_business_statistics(businesses, reviews, tips, checkins)
    keyword_map = top_keywords_per_business(reviews)
    _log_stage("Aggregating business statistics and keywords", stage_started_at)

    print("[pipeline] Discovering latent topics and embeddings...")
    stage_started_at = time.perf_counter()
    topic_result = discover_topics(
        reviews=reviews,
        embedding_model_name=config.embedding_model,
        min_topic_size=config.min_topic_size,
        examples_per_topic=config.representation_docs_per_topic,
    )
    print(f"[pipeline] Topic discovery complete. Found {len(topic_result.topic_keywords):,} topics.")
    _log_stage("Discovering latent topics and embeddings", stage_started_at)

    print("[pipeline] Labeling attributes and computing similarities...")
    stage_started_at = time.perf_counter()
    attributes = build_attribute_records(
        topic_result.topic_keywords,
        topic_result.topic_examples,
        model_name=config.openai_model if config.use_openai_labels else None,
    )
    attribute_sentiment = summarize_attribute_sentiment(topic_result.assignments)
    attribute_sentiment_lookup = {row["attribute_id"]: row for row in attribute_sentiment}
    for record in attributes:
        record.update(attribute_sentiment_lookup.get(record["attribute_id"], {}))
    business_attributes = summarize_business_attributes(topic_result.assignments, config.top_n_topics)
    attribute_similarity = attribute_similarity_edges(attributes)
    business_sentiment = summarize_business_sentiment(topic_result.assignments)
    if business_sentiment:
        business_stats = business_stats.merge(pd.DataFrame(business_sentiment), how="left", on="business_id")
    for col in ["avg_sentiment", "positive_share", "neutral_share", "negative_share"]:
        if col not in business_stats.columns:
            business_stats[col] = 0.0
        business_stats[col] = business_stats[col].fillna(0.0)
    _log_stage("Labeling attributes and computing similarities", stage_started_at)

    print("[pipeline] Building and exporting graph...")
    stage_started_at = time.perf_counter()
    graph = build_graph(
        business_stats=business_stats,
        attributes=attributes,
        business_attributes=business_attributes,
        keyword_map=keyword_map,
        attribute_similarity=attribute_similarity,
    )
    export_graph(graph, config.output_dir)
    _log_stage("Building and exporting graph", stage_started_at)

    print("[pipeline] Writing tabular and JSON artifacts...")
    stage_started_at = time.perf_counter()
    reviews.to_csv(config.output_dir / "sampled_reviews.csv", index=False)
    topic_result.assignments.to_csv(config.output_dir / "topic_assignments.csv", index=False)
    _write_json(config.output_dir / "attributes.json", attributes)
    _write_json(config.output_dir / "business_attributes.json", business_attributes)
    _write_json(config.output_dir / "business_sentiment.json", business_sentiment)
    _log_stage("Writing tabular and JSON artifacts", stage_started_at)

    attribute_lookup = {row["attribute_id"]: row["label"] for row in attributes}
    business_rows = []
    for row in business_stats.itertuples(index=False):
        attr_labels = [
            attribute_lookup[item["attribute_id"]]
            for item in business_attributes
            if item["business_id"] == row.business_id and item["attribute_id"] in attribute_lookup
        ]
        search_text = " | ".join(
            part
            for part in [
                row.name or "",
                row.city or "",
                row.categories or "",
                ", ".join(attr_labels),
                f"sentiment {round(float(row.avg_sentiment), 4)}",
                ", ".join(keyword_map.get(row.business_id, [])),
            ]
            if part
        )
        business_rows.append(
            {
                "business_id": row.business_id,
                "name": row.name,
                "city": row.city,
                "categories": row.categories,
                "attributes": attr_labels,
                "avg_sentiment": round(float(row.avg_sentiment), 4),
                "search_text": search_text,
            }
        )

    print("[pipeline] Building semantic query index...")
    stage_started_at = time.perf_counter()
    build_query_index(config.output_dir, business_rows, config.embedding_model)
    _log_stage("Building semantic query index", stage_started_at)

    summary = {
        "businesses_in_scope": int(len(businesses)),
        "sampled_reviews": int(len(reviews)),
        "discovered_attributes": int(len(attributes)),
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "avg_sampled_sentiment": round(float(business_stats["avg_sentiment"].mean()), 4),
        "used_bertopic": topic_result.used_bertopic,
        "embedding_model": topic_result.embedding_model_name,
        "use_all_reviews": config.use_all_reviews,
        "state_filter": config.state_filter,
        "city_filter": config.city_filter,
        "category_filter": config.category_filter,
        "output_dir": str(config.output_dir),
    }
    _write_json(config.output_dir / "summary.json", summary)
    print(f"[pipeline] Run complete in {_format_elapsed(time.perf_counter() - total_started_at)}.")
    return summary
