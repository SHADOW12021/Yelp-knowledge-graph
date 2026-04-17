from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pandas as pd


def build_graph(
    business_stats: pd.DataFrame,
    attributes: list[dict],
    business_attributes: list[dict],
    keyword_map: dict[str, list[str]],
    attribute_similarity: list[dict],
) -> nx.Graph:
    graph = nx.Graph()

    for row in business_stats.itertuples(index=False):
        graph.add_node(
            row.business_id,
            node_type="business",
            name=row.name,
            city=row.city,
            state=row.state,
            stars=float(row.stars) if row.stars is not None else 0.0,
            review_count=int(row.review_count),
            sampled_review_count=int(row.sampled_review_count),
            tip_count=int(row.tip_count),
            checkin_count=int(row.checkin_count),
            categories=row.categories or "",
        )
        if row.city:
            city_id = f"city::{row.city}"
            graph.add_node(city_id, node_type="city", label=row.city)
            graph.add_edge(row.business_id, city_id, relation="located_in")

        for category in [part.strip() for part in str(row.categories).split(",") if part.strip()]:
            category_id = f"category::{category}"
            graph.add_node(category_id, node_type="category", label=category)
            graph.add_edge(row.business_id, category_id, relation="has_category")

    for attr in attributes:
        graph.add_node(
            attr["attribute_id"],
            node_type="attribute",
            label=attr["label"],
            topic_id=int(attr["topic_id"]),
        )
        for keyword in attr["keywords"]:
            keyword_id = f"keyword::{keyword}"
            graph.add_node(keyword_id, node_type="keyword", label=keyword)
            graph.add_edge(attr["attribute_id"], keyword_id, relation="described_by")

    for row in business_attributes:
        graph.add_edge(
            row["business_id"],
            row["attribute_id"],
            relation="has_attribute",
            strength=float(row["strength"]),
            review_count=int(row["review_count"]),
        )

    for business_id, keywords in keyword_map.items():
        for keyword in keywords:
            keyword_id = f"keyword::{keyword}"
            if graph.has_node(business_id):
                graph.add_node(keyword_id, node_type="keyword", label=keyword)
                graph.add_edge(business_id, keyword_id, relation="mentions_keyword")

    for row in attribute_similarity:
        graph.add_edge(
            row["source"],
            row["target"],
            relation="similar_attribute",
            similarity=float(row["similarity"]),
        )

    return graph


def export_graph(graph: nx.Graph, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    graphml_path = output_dir / "graph.graphml"
    nx.write_graphml(graph, graphml_path)

    payload = {
        "nodes": [{"id": node_id, **attrs} for node_id, attrs in graph.nodes(data=True)],
        "edges": [{"source": u, "target": v, **attrs} for u, v, attrs in graph.edges(data=True)],
    }
    with (output_dir / "graph.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
