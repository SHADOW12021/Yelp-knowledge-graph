from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yelp_kg.device import build_device_banner, detect_torch_device


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in relevant_ids)
    return hits / k


def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for index, item in enumerate(ranked_ids, start=1):
        if item in relevant_ids:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    top = ranked_ids[:k]
    dcg = 0.0
    for index, item in enumerate(top, start=1):
        rel = 1.0 if item in relevant_ids else 0.0
        if rel > 0:
            dcg += rel / math.log2(index + 1)
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def category_exact_scores(query: str, categories: list[str]) -> np.ndarray:
    query_tokens = {token.strip().lower() for token in query.replace("-", " ").split() if token.strip()}
    scores = []
    for category_text in categories:
        category_tokens = {token.strip().lower() for token in category_text.replace(",", " ").replace("&", " ").split() if token.strip()}
        overlap = len(query_tokens & category_tokens)
        scores.append(float(overlap))
    return np.array(scores, dtype=float)


def compute_metrics(rankings: dict[str, list[str]], qrels: dict[str, set[str]]) -> dict[str, float]:
    p5 = []
    p10 = []
    ndcg10 = []
    mrr_scores = []
    for query, ranked_ids in rankings.items():
        relevant_ids = qrels.get(query, set())
        p5.append(precision_at_k(ranked_ids, relevant_ids, 5))
        p10.append(precision_at_k(ranked_ids, relevant_ids, 10))
        ndcg10.append(ndcg_at_k(ranked_ids, relevant_ids, 10))
        mrr_scores.append(mrr(ranked_ids, relevant_ids))
    return {
        "precision_at_5": round(float(np.mean(p5)), 4),
        "precision_at_10": round(float(np.mean(p10)), 4),
        "ndcg_at_10": round(float(np.mean(ndcg10)), 4),
        "mrr": round(float(np.mean(mrr_scores)), 4),
    }


def build_qrels(queries: list[dict], rows: list[dict]) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    for query_record in queries:
        query = query_record["query"]
        targets = [item.lower() for item in query_record["target_categories"]]
        relevant = set()
        for row in rows:
            categories = str(row.get("categories") or "").lower()
            if any(target in categories for target in targets):
                relevant.add(row["business_id"])
        qrels[query] = relevant
    return qrels


def rank_from_scores(scores: np.ndarray, business_ids: list[str]) -> list[str]:
    order = np.argsort(scores)[::-1]
    return [business_ids[idx] for idx in order]


def evaluate(
    artifacts_dir: Path,
    queries_path: Path,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_json(artifacts_dir / "query_index.json")
    queries = _load_json(queries_path)
    rows = payload["rows"]

    business_ids = [row["business_id"] for row in rows]
    category_docs = [str(row.get("categories") or "") for row in rows]
    lexical_docs = [
        " | ".join(
            [
                str(row.get("name") or ""),
                str(row.get("city") or ""),
                str(row.get("categories") or ""),
            ]
        )
        for row in rows
    ]
    proposed_embeddings = np.array([row["embedding"] for row in rows], dtype=float)

    qrels = build_qrels(queries, rows)

    lexical_vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    lexical_matrix = lexical_vectorizer.fit_transform(lexical_docs)

    from sentence_transformers import SentenceTransformer

    device_info = detect_torch_device()
    print(build_device_banner("evaluation"))
    embedder = SentenceTransformer(payload["embedding_model"], device=device_info.device)

    no_attribute_texts = lexical_docs
    baseline_embeddings = embedder.encode(no_attribute_texts, show_progress_bar=True, normalize_embeddings=True)

    rankings: dict[str, dict[str, list[str]]] = {
        "category_exact_match": {},
        "tfidf_business_text": {},
        "embedding_without_latent_attributes": {},
        "proposed_embedding_with_latent_attributes": {},
    }
    per_query_rows: list[dict] = []

    for query_record in queries:
        query = query_record["query"]
        relevant_ids = qrels[query]

        exact_scores = category_exact_scores(query, category_docs)
        rankings["category_exact_match"][query] = rank_from_scores(exact_scores, business_ids)

        lexical_query = lexical_vectorizer.transform([query])
        lexical_scores = (lexical_matrix @ lexical_query.T).toarray().ravel()
        rankings["tfidf_business_text"][query] = rank_from_scores(lexical_scores, business_ids)

        query_embedding = embedder.encode([query], normalize_embeddings=True)[0]
        baseline_scores = baseline_embeddings @ query_embedding
        rankings["embedding_without_latent_attributes"][query] = rank_from_scores(baseline_scores, business_ids)

        proposed_scores = proposed_embeddings @ query_embedding
        rankings["proposed_embedding_with_latent_attributes"][query] = rank_from_scores(proposed_scores, business_ids)

        per_query_rows.append(
            {
                "query": query,
                "relevant_businesses": len(relevant_ids),
                "category_exact_match_p5": round(precision_at_k(rankings["category_exact_match"][query], relevant_ids, 5), 4),
                "tfidf_business_text_p5": round(precision_at_k(rankings["tfidf_business_text"][query], relevant_ids, 5), 4),
                "embedding_without_latent_attributes_p5": round(precision_at_k(rankings["embedding_without_latent_attributes"][query], relevant_ids, 5), 4),
                "proposed_embedding_with_latent_attributes_p5": round(precision_at_k(rankings["proposed_embedding_with_latent_attributes"][query], relevant_ids, 5), 4),
            }
        )

    metrics = {method: compute_metrics(method_rankings, qrels) for method, method_rankings in rankings.items()}
    metrics["benchmark_notes"] = {
        "type": "weakly_supervised_category_grounded_retrieval",
        "queries_file": str(queries_path),
        "artifacts_dir": str(artifacts_dir),
        "business_count": len(rows),
        "query_count": len(queries),
        "relevance_definition": "A business is treated as relevant when its Yelp categories contain one of the query's target categories.",
    }

    pd.DataFrame(per_query_rows).to_csv(output_dir / "per_query_metrics.csv", index=False)
    with (output_dir / "retrieval_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=True)

    top_examples: list[dict] = []
    by_id = {row["business_id"]: row for row in rows}
    for query_record in queries[:5]:
        query = query_record["query"]
        top_ids = rankings["proposed_embedding_with_latent_attributes"][query][:5]
        top_examples.append(
            {
                "query": query,
                "top_results": [
                    {
                        "business_id": business_id,
                        "name": by_id[business_id]["name"],
                        "city": by_id[business_id]["city"],
                        "categories": by_id[business_id]["categories"],
                        "attributes": by_id[business_id]["attributes"][:5],
                    }
                    for business_id in top_ids
                ],
            }
        )
    with (output_dir / "top_result_examples.json").open("w", encoding="utf-8") as handle:
        json.dump(top_examples, handle, indent=2, ensure_ascii=True)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval baselines for the Yelp latent attribute pipeline.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts") / "pa_full_reviews",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=Path("evaluation") / "pa_queries.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "evaluation",
    )
    args = parser.parse_args()

    metrics = evaluate(args.artifacts_dir, args.queries_file, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
