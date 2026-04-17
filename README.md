# Yelp Latent Attribute Knowledge Graph

This project turns the Yelp academic JSON dataset into a structured knowledge system:

- review text -> semantic embeddings
- embeddings -> latent attribute clusters
- clusters -> human-readable labels
- labels + businesses + keywords -> knowledge graph
- graph -> semantic querying

It is designed for the JSONL files already present in this repo under `Yelp-JSON\Yelp JSON\yelp_dataset`.

## What This Implements

The pipeline does five things:

1. Streams the Yelp JSONL files without loading the whole dataset into memory.
2. Samples and filters review text for businesses you care about.
3. Discovers hidden business attributes using `SentenceTransformer` embeddings and `BERTopic`.
4. Produces readable labels for those attributes using keywords plus representative review snippets.
5. Builds a graph linking businesses, attributes, keywords, cities, categories, and check-in signals.

## Project Layout

```text
src/yelp_kg/
  cli.py
  config.py
  data.py
  topic_modeling.py
  labeling.py
  graph_builder.py
  query.py
  pipeline.py
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

## Run The Full Pipeline

This command uses the dataset already in the repo and writes outputs into `artifacts\demo_run`.

```powershell
yelp-kg run `
  --dataset-dir "Yelp-JSON\Yelp JSON\yelp_dataset" `
  --output-dir "artifacts\demo_run" `
  --sample-size 20000 `
  --min-business-reviews 20 `
  --min-topic-size 25 `
  --top-n-topics 20
```

You can also use the included local runner:

```powershell
$env:PYTHONPATH="src"
python scripts\run_local_pipeline.py
```

## Ask Semantic Questions

After the pipeline runs, you can query for businesses that match abstract preferences:

```powershell
yelp-kg query `
  --artifacts-dir "artifacts\demo_run" `
  --text "quiet places to work with coffee and wifi" `
  --top-k 10
```

## Outputs

The pipeline writes:

- `sampled_reviews.csv`: sampled review subset used for modeling
- `topic_assignments.csv`: review-to-topic assignments
- `attributes.json`: discovered latent attributes with labels, keywords, and example reviews
- `business_attributes.json`: business-to-attribute strengths
- `graph.graphml`: knowledge graph for graph tools
- `graph.json`: JSON export of graph nodes and edges
- `query_index.json`: compact search index for semantic querying
- `summary.json`: run metadata and quick stats

## Notes

- The full Yelp review file is several GB, so the default workflow samples reviews instead of embedding everything.
- The code uses BERTopic when available. If BERTopic cannot be imported, it falls back to a TF-IDF + KMeans baseline so the project remains runnable.
- Optional OpenAI labeling is supported through `OPENAI_API_KEY`, but the default implementation already produces readable labels without it.

## Suggested Next Steps

- Restrict to a city or category for a cleaner project story.
- Add Neo4j export if your class wants graph database demos.
- Compare latent attributes against Yelp metadata such as `categories`, `stars`, and `attributes`.
