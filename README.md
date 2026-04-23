# Yelp Latent Attribute Knowledge Graph

This project turns Yelp review text into a structured, queryable knowledge system. Instead of relying only on Yelp's explicit metadata, it discovers hidden business traits from review language, such as:

- `quiet study-friendly space`
- `nightlife and bar scene`
- `romantic date-night atmosphere`
- `service quality and staff experience`
- `coffee-focused casual hangout`

The full pipeline is:

- review text -> semantic embeddings
- embeddings -> latent attribute clusters
- clusters -> human-readable labels
- labels + review sentiment -> richer attribute profiles
- labels + businesses + keywords -> knowledge graph
- graph -> semantic querying and interactive visualization

It is designed for the Yelp academic JSONL files under `Yelp-JSON\Yelp JSON\yelp_dataset`.

## What This Project Does

The system reads Yelp business and review data, embeds reviews into semantic vectors using Sentence-Transformers, clusters semantically similar reviews with BERTopic, and treats each discovered cluster as a latent attribute. A latent attribute is an implicit business quality or theme that is not manually labeled in the dataset but appears consistently in user reviews.

For example, many reviews might mention words like `quiet`, `wifi`, `coffee`, `work`, and `study`. Even if Yelp does not explicitly tag those businesses that way, the model can group those reviews together and generate an interpretable attribute like `quiet study-friendly space`. The pipeline then connects businesses to the attributes that appear most strongly in their reviews, adds sentiment summaries, and exports everything as a knowledge graph plus a semantic search index.

## Project Layout

```text
src/yelp_kg/
  app.py
  cli.py
  config.py
  data.py
  device.py
  graph_builder.py
  labeling.py
  pipeline.py
  query.py
  sentiment.py
  topic_modeling.py
scripts/
  preliminary_discovery.py
  run_local_pipeline.py
```

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

If you want optional LLM-based labeling:

```powershell
pip install -e .[llm]
```

## Run The Pipeline

This command runs the pipeline on Pennsylvania businesses only and uses all matching reviews:

```powershell
yelp-kg run `
  --dataset-dir "Yelp-JSON\Yelp JSON\yelp_dataset" `
  --output-dir "artifacts\pa_full_reviews" `
  --state-filter "PA" `
  --use-all-reviews
```

You can also use the included local runner:

```powershell
$env:PYTHONPATH="src"
python scripts\run_local_pipeline.py
```

## Download Prebuilt PA Artifacts

Prebuilt Pennsylvania artifacts can be downloaded here:

https://drive.google.com/file/d/1v--0OU0yJUhUzyoahQrYVM63FLTVFkul/view?usp=sharing

After downloading:

1. Unzip the downloaded file into the `artifacts` folder.
2. Make sure the final path is `artifacts\pa_full_reviews`.
3. The app expects files such as `graph.json`, `attributes.json`, `business_attributes.json`, `business_sentiment.json`, and `query_index.json` to be inside `artifacts\pa_full_reviews`.

If you unzip it somewhere else, the app command will not find the generated artifacts.

## Ask Semantic Questions

After the pipeline runs, query businesses by abstract preference:

```powershell
yelp-kg query `
  --artifacts-dir "artifacts\pa_full_reviews" `
  --text "quiet places to work with coffee and wifi" `
  --top-k 10
```

## Launch The App

Launch the interactive explorer on Pennsylvania:

```powershell
yelp-kg app `
  --artifacts-dir "artifacts\pa_full_reviews" `
  --host "127.0.0.1" `
  --port 7860
```

The app includes:

- semantic search for businesses
- sentiment-aware business summaries
- interactive knowledge graph visualization around a selected business

## How Latent Attributes Are Created

Latent attributes are discovered automatically from the review text. The pipeline does not start with labels like `romantic`, `study-friendly`, or `nightlife spot`. Instead, it creates them in stages:

1. Reviews are filtered to the businesses in scope, such as only `PA` businesses.
2. Each review is embedded into a dense semantic vector using `SentenceTransformer`.
3. BERTopic groups semantically similar reviews into clusters.
4. Each cluster becomes a candidate latent attribute.
5. The cluster is summarized using top keywords and representative review examples.
6. A readable label is generated heuristically, or optionally with an LLM.
7. Businesses are linked to the attributes that dominate their reviews.

This means a latent attribute is not a supervised class. It is a hidden semantic theme learned from recurring review language.

Example patterns:

- Reviews mentioning `quiet`, `wifi`, `study`, `work`, `coffee` -> `quiet study-friendly space`
- Reviews mentioning `bar`, `music`, `crowded`, `drinks`, `late` -> `nightlife and bar scene`
- Reviews mentioning `service`, `staff`, `friendly`, `attentive` -> `service quality and staff experience`
- Reviews mentioning `romantic`, `date`, `ambience`, `dinner` -> `romantic date-night atmosphere`

## Generated Data And What It Means

The pipeline writes the following artifacts into the output directory, typically `artifacts\pa_full_reviews`.

- `sampled_reviews.csv`
  Review-level dataset actually used by the modeling stage. When `--use-all-reviews` is enabled, this contains all matching reviews even though the filename still says `sampled`.
- `topic_assignments.csv`
  Review-level assignments showing which latent topic each review belongs to.
- `attributes.json`
  The discovered latent attributes. Each record contains an `attribute_id`, the readable `label`, top `keywords`, example reviews, and sentiment summaries for that attribute.
- `business_attributes.json`
  Links each business to its strongest latent attributes. Includes `strength`, which reflects how dominant the attribute is within that business's modeled reviews.
- `business_sentiment.json`
  Business-level sentiment summary computed from review sentiment scores. Includes average sentiment and positive/neutral/negative shares.
- `graph.graphml`
  Graph export for graph tools such as Gephi or Neo4j import workflows.
- `graph.json`
  JSON graph export used by the interactive app. Contains nodes and edges for businesses, attributes, categories, keywords, and cities.
- `query_index.json`
  Embedding-based semantic retrieval index used by the `query` command and the app.
- `summary.json`
  High-level run metadata such as number of businesses, number of reviews, number of discovered attributes, graph size, filters used, and runtime context.

## What The App Table Means

The `Semantic matches` table in the app is created in [src/yelp_kg/app.py](C:/Users/gsocc/OneDrive/Desktop/NLP/Final%20Project/src/yelp_kg/app.py:1). Its columns mean:

- `business`
  The Yelp business name returned by semantic search.
- `city`
  The city where the business is located.
- `score`
  Semantic similarity between the user query and the business search representation.
  Higher is better.
  Lower means the business is less semantically aligned with the query.
  This is mainly a ranking score, not a probability.
  In this project it is based on cosine similarity between the query embedding and the business embedding.
- `avg_sentiment`
  Average VADER compound sentiment for the modeled reviews of that business.
  This usually ranges from `-1` to `1`.
  Higher means reviews are more positive on average.
  Lower means reviews are more negative on average.
  Around `0` means mixed or neutral sentiment.
- `attributes`
  A short list of the strongest discovered latent attributes for that business.
- `categories`
  Yelp's original category string for the business.
- `business_id`
  The unique Yelp business identifier.

Interpretation tips:

- High `score` + high `avg_sentiment`
  The business is a strong semantic match to the query and is discussed positively.
- High `score` + low `avg_sentiment`
  The business matches the concept the user asked for, but reviews are more negative or mixed.
- Low `score`
  The business is likely not a strong conceptual match for the query, even if it has good sentiment.

## Preliminary Discovery Outputs

The exploratory script `scripts\preliminary_discovery.py` creates descriptive statistics before the full latent-attribute pipeline. Useful files include:

- `businesses_per_state.csv`
  Number of businesses per state.
- `top_cities_by_business_count.csv`
  Cities with the largest number of businesses.
- `top_categories.csv`
  Most common Yelp business categories.
- `reviews_per_business.csv`
  Actual review counts per business from the review file.
- `top_reviewed_businesses.csv`
  Businesses with the highest review volume.
- `reviews_per_year.csv`
  Review activity over time.
- `review_star_distribution.csv`
  Distribution of review ratings.
- `users_by_join_year.csv`
  Distribution of user account creation years.

These files are useful for motivating why we focused on Pennsylvania and for describing the dataset in a presentation.

## Run Evaluation

The repository now includes a retrieval evaluation script and a weakly supervised Pennsylvania query set:

- Query set: [evaluation/pa_queries.json](C:/Users/gsocc/OneDrive/Desktop/NLP/Final%20Project/evaluation/pa_queries.json:1)
- Evaluation script: [scripts/evaluate_baselines.py](C:/Users/gsocc/OneDrive/Desktop/NLP/Final%20Project/scripts/evaluate_baselines.py:1)

Run it with:

```powershell
.venv\Scripts\python scripts\evaluate_baselines.py `
  --artifacts-dir "artifacts\pa_full_reviews" `
  --queries-file "evaluation\pa_queries.json" `
  --output-dir "artifacts\evaluation"
```

It writes:

- `artifacts\evaluation\retrieval_metrics.json`
- `artifacts\evaluation\per_query_metrics.csv`
- `artifacts\evaluation\top_result_examples.json`

## Notes

- The full Yelp review file is several GB, so sampling is the default behavior unless `--use-all-reviews` is enabled.
- Use `--use-all-reviews` if you want every matching review instead of sampling.
- The code now reports whether embeddings are running on `cpu` or `cuda`.
- BERTopic is the main topic-modeling method. If BERTopic cannot be imported, the code falls back to a TF-IDF + KMeans baseline so the system remains runnable.
- Optional OpenAI labeling is supported through `OPENAI_API_KEY`, but the default heuristic labeling already produces readable attribute names.

## Slide-Ready Presentation Bullets

### Proposed Research Question

- Can we automatically discover interpretable business attributes from Yelp review text without predefined labels?
- Can those discovered attributes improve semantic retrieval beyond using only Yelp categories or keyword search?
- Can a knowledge graph built from discovered attributes explain why a business matches an abstract user preference?

### Proposed Model

- Review encoder: `SentenceTransformer` (`all-MiniLM-L6-v2`)
- Topic discovery: `BERTopic`
- Attribute labeling: keyword-based heuristic labeling, with optional LLM labeling
- Sentiment layer: VADER sentiment scoring
- Knowledge representation: graph linking businesses, attributes, keywords, categories, and cities
- Retrieval: embedding-based semantic search over business representations enriched with latent attributes

### Algorithm Details

- Stream Yelp JSONL files and filter to the target slice, such as `PA`
- Encode reviews into semantic vectors
- Cluster reviews into latent topics with BERTopic
- Convert topics into latent attributes using top keywords and exemplar reviews
- Aggregate attribute strength per business
- Score review sentiment and aggregate it to businesses and attributes
- Export a knowledge graph and semantic query index
- Use the graph and index in an interactive app for retrieval and explanation

### Dataset Description

- Source: Yelp academic dataset
- Core files used: businesses, reviews, tips, check-ins, users
- Main text signal: review text
- Structured business context: categories, city, state, stars, review counts
- Final presentation subset: Pennsylvania businesses and their reviews
- Motivation for PA subset: high business and review density, faster runtime, cleaner presentation scope

### Evaluation Results Including Baselines

Implemented evaluation:

- Benchmark type:
  Weakly supervised category-grounded retrieval benchmark.
- Query set size:
  15 Pennsylvania-focused natural-language queries.
- Relevance definition:
  A business is treated as relevant if its Yelp categories contain one of the target categories associated with the query.
- Metrics:
  Precision@5, Precision@10, nDCG@10, and Mean Reciprocal Rank.

Baselines evaluated:

- `category_exact_match`
  Very simple lexical overlap against category text.
- `tfidf_business_text`
  TF-IDF retrieval using plain business text: name, city, and categories.
- `embedding_without_latent_attributes`
  Embedding retrieval using name, city, and categories only.
- `proposed_embedding_with_latent_attributes`
  Embedding retrieval using the full semantic business representation, including discovered latent attributes.

Measured results on `artifacts\pa_full_reviews`:

| Method | Precision@5 | Precision@10 | nDCG@10 | MRR |
|---|---:|---:|---:|---:|
| `category_exact_match` | 0.8800 | 0.8667 | 0.8694 | 0.9000 |
| `tfidf_business_text` | 0.8800 | 0.8867 | 0.8899 | 0.9667 |
| `embedding_without_latent_attributes` | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `proposed_embedding_with_latent_attributes` | 1.0000 | 0.9800 | 0.9865 | 1.0000 |

What these results mean:

- The embedding-based methods clearly outperform the lexical baselines on this benchmark.
- The proposed latent-attribute system performs extremely well, but on this particular benchmark it is slightly below the plain embedding baseline.
- That does not necessarily mean latent attributes are worse.
  This benchmark defines relevance using explicit Yelp categories, so it naturally favors methods that match category information very directly.
- In other words, this evaluation is good for measuring category-grounded retrieval, but it likely undervalues the main research contribution of the project:
  abstract semantic matching and interpretable attribute discovery.

Important interpretation:

- A benchmark based on category labels is useful as a first automatic evaluation.
- It is not sufficient to fully test whether latent attributes help with open-ended concepts like `study-friendly`, `romantic atmosphere`, or `nightlife energy`.
- For that reason, the next best evaluation is a small human-judged query set where relevance is based on actual user intent, not only category overlap.

Recommended next evaluation step:

- Build 20 to 40 abstract queries such as `quiet place to work`, `good date night atmosphere`, `fun bar scene`, `family-friendly brunch`.
- Manually judge the top 10 results from each method.
- Compare the same metrics again.
- Also compare explanation quality by judging whether the returned latent attributes actually justify the ranking.

### Conclusion And Limitations

- The pipeline can turn raw review text into interpretable latent business attributes without manual labels.
- The knowledge graph makes semantic search more explainable because businesses are connected to attributes and keywords.
- Restricting to Pennsylvania makes the system easier to run and present while keeping a large enough dataset slice.
- Current limitations:
  labels are heuristic by default, BERTopic quality depends on the review subset, sentiment is lexicon-based, and the system has not yet been formally evaluated against baselines.
- Future improvements:
  add formal evaluation, better labeling with LLMs, graph database export, temporal analysis, and checkpointing for long runs.
