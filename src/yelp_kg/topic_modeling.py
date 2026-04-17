from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(slots=True)
class TopicModelResult:
    assignments: pd.DataFrame
    topic_keywords: dict[int, list[tuple[str, float]]]
    topic_examples: dict[int, list[str]]
    embedding_model_name: str
    used_bertopic: bool


def _fit_bertopic(
    texts: list[str],
    embeddings: np.ndarray,
    min_topic_size: int,
) -> tuple[list[int], dict[int, list[tuple[str, float]]], bool]:
    from bertopic import BERTopic

    model = BERTopic(
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(texts, embeddings=embeddings)
    topic_keywords: dict[int, list[tuple[str, float]]] = {}
    for topic_id in set(topics):
        if topic_id == -1:
            continue
        topic_keywords[topic_id] = model.get_topic(topic_id) or []
    return topics, topic_keywords, True


def _fit_fallback(texts: list[str], min_topic_size: int) -> tuple[list[int], dict[int, list[tuple[str, float]]], bool]:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=3000,
        ngram_range=(1, 2),
        min_df=5,
    )
    matrix = vectorizer.fit_transform(texts)
    n_clusters = max(2, min(20, len(texts) // max(min_topic_size, 5)))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    topics = model.fit_predict(matrix)

    feature_names = vectorizer.get_feature_names_out()
    topic_keywords: dict[int, list[tuple[str, float]]] = {}
    order = model.cluster_centers_.argsort(axis=1)[:, ::-1]
    for topic_id in range(n_clusters):
        topic_keywords[topic_id] = [
            (feature_names[idx], float(model.cluster_centers_[topic_id, idx]))
            for idx in order[topic_id][:10]
        ]
    return topics.tolist(), topic_keywords, False


def discover_topics(
    reviews: pd.DataFrame,
    embedding_model_name: str,
    min_topic_size: int,
    examples_per_topic: int,
) -> TopicModelResult:
    from sentence_transformers import SentenceTransformer

    texts = reviews["text"].tolist()
    embedder = SentenceTransformer(embedding_model_name)
    embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    try:
        topics, topic_keywords, used_bertopic = _fit_bertopic(texts, embeddings, min_topic_size)
    except Exception:
        topics, topic_keywords, used_bertopic = _fit_fallback(texts, min_topic_size)

    assignments = reviews.copy()
    assignments["topic_id"] = topics

    topic_examples: dict[int, list[str]] = {}
    for topic_id, group in assignments.groupby("topic_id"):
        if topic_id == -1:
            continue
        ranked = group.sort_values(["useful", "stars"], ascending=[False, False]).head(examples_per_topic)
        topic_examples[int(topic_id)] = ranked["text"].tolist()

    return TopicModelResult(
        assignments=assignments,
        topic_keywords={int(k): v for k, v in topic_keywords.items()},
        topic_examples=topic_examples,
        embedding_model_name=embedding_model_name,
        used_bertopic=used_bertopic,
    )
