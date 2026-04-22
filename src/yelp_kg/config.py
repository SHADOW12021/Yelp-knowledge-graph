from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DatasetPaths:
    dataset_dir: Path

    @property
    def business_path(self) -> Path:
        return self.dataset_dir / "yelp_academic_dataset_business.json"

    @property
    def review_path(self) -> Path:
        return self.dataset_dir / "yelp_academic_dataset_review.json"

    @property
    def tip_path(self) -> Path:
        return self.dataset_dir / "yelp_academic_dataset_tip.json"

    @property
    def checkin_path(self) -> Path:
        return self.dataset_dir / "yelp_academic_dataset_checkin.json"

    @property
    def user_path(self) -> Path:
        return self.dataset_dir / "yelp_academic_dataset_user.json"


@dataclass(slots=True)
class PipelineConfig:
    dataset_dir: Path
    output_dir: Path
    sample_size: int = 20_000
    use_all_reviews: bool = False
    random_seed: int = 42
    state_filter: str | None = None
    city_filter: str | None = None
    category_filter: str | None = None
    min_business_reviews: int = 20
    min_review_length: int = 40
    min_topic_size: int = 25
    top_n_topics: int = 20
    embedding_model: str = "all-MiniLM-L6-v2"
    representation_docs_per_topic: int = 5
    use_openai_labels: bool = False
    openai_model: str = "gpt-4o-mini"

    @property
    def paths(self) -> DatasetPaths:
        return DatasetPaths(self.dataset_dir)
