from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from yelp_kg.config import DatasetPaths
from yelp_kg.data import stream_jsonl


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def _safe_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _series_stats(series: pd.Series) -> dict:
    if series.empty:
        return {}
    return {
        "count": int(series.count()),
        "mean": _safe_float(series.mean()),
        "median": _safe_float(series.median()),
        "min": _safe_float(series.min()),
        "max": _safe_float(series.max()),
        "p90": _safe_float(series.quantile(0.90)),
        "p95": _safe_float(series.quantile(0.95)),
        "p99": _safe_float(series.quantile(0.99)),
    }


def analyze_businesses(paths: DatasetPaths, output_dir: Path, top_k: int) -> pd.DataFrame:
    rows: list[dict] = []
    state_counter: Counter[str] = Counter()
    city_counter: Counter[tuple[str, str]] = Counter()
    category_counter: Counter[str] = Counter()

    for row in stream_jsonl(paths.business_path):
        state = row.get("state") or "UNKNOWN"
        city = row.get("city") or "UNKNOWN"
        categories = row.get("categories") or ""

        rows.append(
            {
                "business_id": row["business_id"],
                "name": row.get("name"),
                "city": city,
                "state": state,
                "stars": row.get("stars"),
                "review_count": row.get("review_count", 0),
                "is_open": row.get("is_open", 0),
                "categories": categories,
            }
        )
        state_counter[state] += 1
        city_counter[(state, city)] += 1
        for category in [part.strip() for part in categories.split(",") if part.strip()]:
            category_counter[category] += 1

    businesses = pd.DataFrame(rows)

    state_df = pd.DataFrame(
        [{"state": state, "business_count": count} for state, count in state_counter.most_common()]
    )
    city_df = pd.DataFrame(
        [
            {"state": state, "city": city, "business_count": count}
            for (state, city), count in city_counter.most_common(top_k)
        ]
    )
    category_df = pd.DataFrame(
        [{"category": category, "business_count": count} for category, count in category_counter.most_common(top_k)]
    )

    state_df.to_csv(output_dir / "businesses_per_state.csv", index=False)
    city_df.to_csv(output_dir / "top_cities_by_business_count.csv", index=False)
    category_df.to_csv(output_dir / "top_categories.csv", index=False)

    summary = {
        "total_businesses": int(len(businesses)),
        "open_businesses": int((businesses["is_open"] == 1).sum()),
        "closed_businesses": int((businesses["is_open"] == 0).sum()),
        "review_count_stats_from_business_table": _series_stats(businesses["review_count"]),
        "business_star_stats": _series_stats(businesses["stars"].fillna(0)),
        "top_states": state_df.head(top_k).to_dict(orient="records"),
        "top_categories": category_df.head(top_k).to_dict(orient="records"),
    }
    _write_json(output_dir / "business_summary.json", summary)
    return businesses


def analyze_reviews(paths: DatasetPaths, businesses: pd.DataFrame, output_dir: Path, top_k: int) -> None:
    reviews_per_business: Counter[str] = Counter()
    reviews_per_year: Counter[str] = Counter()
    review_star_counter: Counter[int] = Counter()
    review_lengths: list[int] = []
    useful_votes: list[int] = []

    total_reviews = 0
    for row in stream_jsonl(paths.review_path):
        business_id = row.get("business_id")
        date_text = str(row.get("date") or "")
        text = (row.get("text") or "").strip()
        stars = int(row.get("stars", 0) or 0)
        useful = int(row.get("useful", 0) or 0)

        total_reviews += 1
        reviews_per_business[business_id] += 1
        review_star_counter[stars] += 1
        useful_votes.append(useful)
        review_lengths.append(len(text.split()))
        if len(date_text) >= 4:
            reviews_per_year[date_text[:4]] += 1

    review_counts_df = pd.DataFrame(
        [{"business_id": business_id, "actual_review_count": count} for business_id, count in reviews_per_business.items()]
    )
    top_reviewed = (
        review_counts_df.merge(
            businesses[["business_id", "name", "city", "state", "categories"]],
            on="business_id",
            how="left",
        )
        .sort_values("actual_review_count", ascending=False)
        .head(top_k)
    )
    years_df = pd.DataFrame(
        [{"year": year, "review_count": count} for year, count in sorted(reviews_per_year.items())]
    )
    stars_df = pd.DataFrame(
        [{"stars": stars, "review_count": count} for stars, count in sorted(review_star_counter.items())]
    )

    review_counts_df.to_csv(output_dir / "reviews_per_business.csv", index=False)
    top_reviewed.to_csv(output_dir / "top_reviewed_businesses.csv", index=False)
    years_df.to_csv(output_dir / "reviews_per_year.csv", index=False)
    stars_df.to_csv(output_dir / "review_star_distribution.csv", index=False)

    summary = {
        "total_reviews": int(total_reviews),
        "review_count_per_business_stats": _series_stats(review_counts_df["actual_review_count"]),
        "review_length_word_stats": _series_stats(pd.Series(review_lengths)),
        "useful_vote_stats": _series_stats(pd.Series(useful_votes)),
        "top_reviewed_businesses": top_reviewed.to_dict(orient="records"),
    }
    _write_json(output_dir / "review_summary.json", summary)


def analyze_tips(paths: DatasetPaths, output_dir: Path) -> None:
    tip_count = 0
    likes_total = 0
    tip_lengths: list[int] = []

    for row in stream_jsonl(paths.tip_path):
        tip_text = (row.get("text") or "").strip()
        tip_count += 1
        likes_total += int(row.get("likes", 0) or 0)
        tip_lengths.append(len(tip_text.split()))

    summary = {
        "total_tips": int(tip_count),
        "total_tip_likes": int(likes_total),
        "tip_length_word_stats": _series_stats(pd.Series(tip_lengths)),
    }
    _write_json(output_dir / "tip_summary.json", summary)


def analyze_checkins(paths: DatasetPaths, output_dir: Path) -> None:
    business_count = 0
    checkin_counts: list[int] = []

    for row in stream_jsonl(paths.checkin_path):
        date_text = row.get("date") or ""
        count = len([part for part in date_text.split(",") if part.strip()])
        business_count += 1
        checkin_counts.append(count)

    summary = {
        "businesses_with_checkins": int(business_count),
        "checkin_count_stats": _series_stats(pd.Series(checkin_counts)),
    }
    _write_json(output_dir / "checkin_summary.json", summary)


def analyze_users(paths: DatasetPaths, output_dir: Path) -> None:
    user_count = 0
    review_counts: list[int] = []
    fans_counts: list[int] = []
    useful_counts: list[int] = []
    average_stars: list[float] = []
    yelping_years: Counter[str] = Counter()

    for row in stream_jsonl(paths.user_path):
        user_count += 1
        review_counts.append(int(row.get("review_count", 0) or 0))
        fans_counts.append(int(row.get("fans", 0) or 0))
        useful_counts.append(int(row.get("useful", 0) or 0))
        average_stars.append(float(row.get("average_stars", 0) or 0))
        yelping_since = str(row.get("yelping_since") or "")
        if len(yelping_since) >= 4:
            yelping_years[yelping_since[:4]] += 1

    years_df = pd.DataFrame(
        [{"year": year, "user_count": count} for year, count in sorted(yelping_years.items())]
    )
    years_df.to_csv(output_dir / "users_by_join_year.csv", index=False)

    summary = {
        "total_users": int(user_count),
        "user_review_count_stats": _series_stats(pd.Series(review_counts)),
        "user_fan_stats": _series_stats(pd.Series(fans_counts)),
        "user_useful_vote_stats": _series_stats(pd.Series(useful_counts)),
        "user_average_star_stats": _series_stats(pd.Series(average_stars)),
    }
    _write_json(output_dir / "user_summary.json", summary)


def build_overall_summary(output_dir: Path) -> None:
    files = [
        "business_summary.json",
        "review_summary.json",
        "tip_summary.json",
        "checkin_summary.json",
        "user_summary.json",
    ]
    payload = {}
    for name in files:
        path = output_dir / name
        with path.open("r", encoding="utf-8") as handle:
            payload[path.stem] = json.load(handle)
    _write_json(output_dir / "dataset_overview.json", payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run preliminary discovery and exploratory analysis on the Yelp dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("Yelp-JSON") / "Yelp JSON" / "yelp_dataset",
        help="Directory containing the Yelp academic dataset JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "preliminary_discovery",
        help="Directory where discovery outputs will be written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of top states, cities, categories, and businesses to include in summary tables.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = DatasetPaths(args.dataset_dir)
    businesses = analyze_businesses(paths, output_dir, args.top_k)
    analyze_reviews(paths, businesses, output_dir, args.top_k)
    analyze_tips(paths, output_dir)
    analyze_checkins(paths, output_dir)
    analyze_users(paths, output_dir)
    build_overall_summary(output_dir)

    print(f"Preliminary discovery complete. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
