from pathlib import Path

from yelp_kg.config import PipelineConfig
from yelp_kg.pipeline import run_pipeline


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    config = PipelineConfig(
        dataset_dir=root / "Yelp-JSON" / "Yelp JSON" / "yelp_dataset",
        output_dir=root / "artifacts" / "demo_run",
        sample_size=20_000,
        min_business_reviews=20,
        min_topic_size=25,
        top_n_topics=20,
    )
    summary = run_pipeline(config)
    print(summary)


if __name__ == "__main__":
    main()
