from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline
from .query import query_businesses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover latent Yelp business attributes and build a knowledge graph.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the full discovery pipeline.")
    run_parser.add_argument("--dataset-dir", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--sample-size", type=int, default=20_000)
    run_parser.add_argument("--city-filter", type=str, default=None)
    run_parser.add_argument("--category-filter", type=str, default=None)
    run_parser.add_argument("--min-business-reviews", type=int, default=20)
    run_parser.add_argument("--min-review-length", type=int, default=40)
    run_parser.add_argument("--min-topic-size", type=int, default=25)
    run_parser.add_argument("--top-n-topics", type=int, default=20)
    run_parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    run_parser.add_argument("--use-openai-labels", action="store_true")
    run_parser.add_argument("--openai-model", type=str, default="gpt-4o-mini")

    query_parser = subparsers.add_parser("query", help="Query businesses using the exported semantic index.")
    query_parser.add_argument("--artifacts-dir", type=Path, required=True)
    query_parser.add_argument("--text", type=str, required=True)
    query_parser.add_argument("--top-k", type=int, default=10)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        config = PipelineConfig(
            dataset_dir=args.dataset_dir,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            city_filter=args.city_filter,
            category_filter=args.category_filter,
            min_business_reviews=args.min_business_reviews,
            min_review_length=args.min_review_length,
            min_topic_size=args.min_topic_size,
            top_n_topics=args.top_n_topics,
            embedding_model=args.embedding_model,
            use_openai_labels=args.use_openai_labels,
            openai_model=args.openai_model,
        )
        summary = run_pipeline(config)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "query":
        results = query_businesses(args.artifacts_dir, args.text, args.top_k)
        print(json.dumps(results, indent=2))
        return


if __name__ == "__main__":
    main()
