from __future__ import annotations

import json
from pathlib import Path

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
from .topic_modeling import discover_topics


def _write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def run_pipeline(config: PipelineConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    businesses = load_businesses(config)
    if businesses.empty:
        raise ValueError("No businesses matched the current filters.")

    allowed_business_ids = set(businesses["business_id"])
    reviews = reservoir_sample_reviews(config, allowed_business_ids)
    if reviews.empty:
        raise ValueError("No reviews were sampled. Relax the filters or lower review thresholds.")

    tips = load_tips(config.paths, allowed_business_ids)
    checkins = load_checkins(config.paths, allowed_business_ids)
    business_stats = aggregate_business_statistics(businesses, reviews, tips, checkins)
    keyword_map = top_keywords_per_business(reviews)

    topic_result = discover_topics(
        reviews=reviews,
        embedding_model_name=config.embedding_model,
        min_topic_size=config.min_topic_size,
        examples_per_topic=config.representation_docs_per_topic,
    )
    attributes = build_attribute_records(
        topic_result.topic_keywords,
        topic_result.topic_examples,
        model_name=config.openai_model if config.use_openai_labels else None,
    )
    business_attributes = summarize_business_attributes(topic_result.assignments, config.top_n_topics)
    attribute_similarity = attribute_similarity_edges(attributes)

    graph = build_graph(
        business_stats=business_stats,
        attributes=attributes,
        business_attributes=business_attributes,
        keyword_map=keyword_map,
        attribute_similarity=attribute_similarity,
    )
    export_graph(graph, config.output_dir)

    reviews.to_csv(config.output_dir / "sampled_reviews.csv", index=False)
    topic_result.assignments.to_csv(config.output_dir / "topic_assignments.csv", index=False)
    _write_json(config.output_dir / "attributes.json", attributes)
    _write_json(config.output_dir / "business_attributes.json", business_attributes)

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
                "search_text": search_text,
            }
        )
    build_query_index(config.output_dir, business_rows, config.embedding_model)

    summary = {
        "businesses_in_scope": int(len(businesses)),
        "sampled_reviews": int(len(reviews)),
        "discovered_attributes": int(len(attributes)),
        "graph_nodes": int(graph.number_of_nodes()),
        "graph_edges": int(graph.number_of_edges()),
        "used_bertopic": topic_result.used_bertopic,
        "embedding_model": topic_result.embedding_model_name,
        "city_filter": config.city_filter,
        "category_filter": config.category_filter,
        "output_dir": str(config.output_dir),
    }
    _write_json(config.output_dir / "summary.json", summary)
    return summary
