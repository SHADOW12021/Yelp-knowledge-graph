from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .device import build_device_banner, detect_torch_device


def build_query_index(
    output_dir: Path,
    business_rows: list[dict],
    embedding_model_name: str,
) -> None:
    from sentence_transformers import SentenceTransformer

    device_info = detect_torch_device()
    print(build_device_banner("query_index"))

    texts = [row["search_text"] for row in business_rows]
    embedder = SentenceTransformer(embedding_model_name, device=device_info.device)
    embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    payload = {
        "embedding_model": embedding_model_name,
        "rows": [
            {
                **row,
                "embedding": embeddings[idx].tolist(),
            }
            for idx, row in enumerate(business_rows)
        ],
    }
    with (output_dir / "query_index.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True)


def query_businesses(artifacts_dir: Path, text: str, top_k: int) -> list[dict]:
    from sentence_transformers import SentenceTransformer

    device_info = detect_torch_device()
    print(build_device_banner("query"))

    with (artifacts_dir / "query_index.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    embedder = SentenceTransformer(payload["embedding_model"], device=device_info.device)
    query_embedding = embedder.encode([text], normalize_embeddings=True)[0]

    scored: list[dict] = []
    for row in payload["rows"]:
        score = float(np.dot(np.array(row["embedding"], dtype=float), query_embedding))
        scored.append(
            {
                "business_id": row["business_id"],
                "name": row["name"],
                "city": row["city"],
                "score": round(score, 4),
                "attributes": row["attributes"],
                "categories": row["categories"],
                "avg_sentiment": row.get("avg_sentiment"),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]
