from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from .query import query_businesses


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_artifacts(artifacts_dir: Path) -> dict:
    graph_payload = _load_json(artifacts_dir / "graph.json")
    attribute_records = _load_json(artifacts_dir / "attributes.json")
    business_attribute_records = _load_json(artifacts_dir / "business_attributes.json")
    business_sentiment = {
        row["business_id"]: row
        for row in _load_json(artifacts_dir / "business_sentiment.json")
    }

    graph = nx.Graph()
    for node in graph_payload["nodes"]:
        node_id = node["id"]
        attrs = {key: value for key, value in node.items() if key != "id"}
        graph.add_node(node_id, **attrs)
    for edge in graph_payload["edges"]:
        source = edge["source"]
        target = edge["target"]
        attrs = {key: value for key, value in edge.items() if key not in {"source", "target"}}
        graph.add_edge(source, target, **attrs)

    attribute_lookup = {row["attribute_id"]: row for row in attribute_records}
    business_attrs: dict[str, list[dict]] = {}
    for row in business_attribute_records:
        business_attrs.setdefault(row["business_id"], []).append(row)

    return {
        "graph": graph,
        "attributes": attribute_lookup,
        "business_attributes": business_attrs,
        "business_sentiment": business_sentiment,
    }


def render_business_graph(graph: nx.Graph, business_id: str, hops: int = 1) -> go.Figure:
    if business_id not in graph:
        return go.Figure()

    nodes = {business_id}
    frontier = {business_id}
    for _ in range(max(1, hops)):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(graph.neighbors(node))
        nodes.update(next_frontier)
        frontier = next_frontier

    subgraph = graph.subgraph(nodes).copy()
    positions = nx.spring_layout(subgraph, seed=42)

    edge_x: list[float] = []
    edge_y: list[float] = []
    for source, target in subgraph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 1, "color": "#A37B45"},
        hoverinfo="none",
    )

    color_map = {
        "business": "#B22222",
        "attribute": "#0D6E6E",
        "keyword": "#C77D2B",
        "city": "#3B5B92",
        "category": "#556B2F",
    }

    node_x: list[float] = []
    node_y: list[float] = []
    node_text: list[str] = []
    node_color: list[str] = []
    node_size: list[int] = []
    for node_id, attrs in subgraph.nodes(data=True):
        x, y = positions[node_id]
        node_x.append(x)
        node_y.append(y)
        node_type = attrs.get("node_type", "other")
        label = attrs.get("label") or attrs.get("name") or node_id
        node_text.append(f"{label} ({node_type})")
        node_color.append(color_map.get(node_type, "#666666"))
        node_size.append(22 if node_id == business_id else 14)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[text.split(" (")[0] for text in node_text],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker={
            "size": node_size,
            "color": node_color,
            "line": {"width": 1, "color": "#F7F1E5"},
        },
    )

    figure = go.Figure([edge_trace, node_trace])
    figure.update_layout(
        paper_bgcolor="#F7F1E5",
        plot_bgcolor="#F7F1E5",
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False},
        showlegend=False,
    )
    return figure


def launch_app(artifacts_dir: Path) -> gr.Blocks:
    cache = load_artifacts(artifacts_dir)

    def _business_id_from_selection(selection: str | None) -> str | None:
        if not selection:
            return None
        return selection.rsplit("[", 1)[-1].rstrip("]")

    def run_query(query_text: str, top_k: int):
        results = query_businesses(artifacts_dir, query_text, top_k)
        rows = []
        choices = []
        for result in results:
            sentiment = cache["business_sentiment"].get(result["business_id"], {})
            label = f'{result["name"]} ({result["city"]}) [{result["business_id"]}]'
            choices.append(label)
            rows.append(
                {
                    "business": result["name"],
                    "city": result["city"],
                    "score": result["score"],
                    "avg_sentiment": sentiment.get("avg_sentiment"),
                    "attributes": ", ".join(result["attributes"][:5]),
                    "categories": result["categories"],
                    "business_id": result["business_id"],
                }
            )
        dataframe = pd.DataFrame(rows)
        default_choice = choices[0] if choices else None
        graph = render_selected(default_choice) if default_choice else go.Figure()
        details = render_business_details(default_choice)
        return dataframe, gr.update(choices=choices, value=default_choice), graph, details

    def render_selected(selection: str | None):
        business_id = _business_id_from_selection(selection)
        if not business_id:
            return go.Figure()
        return render_business_graph(cache["graph"], business_id)

    def render_business_details(selection: str | None):
        business_id = _business_id_from_selection(selection)
        if not business_id:
            return "No business selected yet."
        attrs = cache["business_attributes"].get(business_id, [])
        sentiment = cache["business_sentiment"].get(business_id, {})
        lines = []
        lines.append(f"Business ID: `{business_id}`")
        if sentiment:
            lines.append(
                f'Sampled sentiment: `{sentiment.get("avg_sentiment")}` '
                f'(positive `{sentiment.get("positive_share")}`, negative `{sentiment.get("negative_share")}`)'
            )
        if attrs:
            lines.append("Top discovered attributes:")
            for item in sorted(attrs, key=lambda row: row["strength"], reverse=True)[:8]:
                attr = cache["attributes"].get(item["attribute_id"], {})
                label = attr.get("label", item["attribute_id"])
                lines.append(
                    f'- {label} | strength `{round(item["strength"], 4)}` | reviews `{item["review_count"]}`'
                )
        return "\n".join(lines)

    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600&display=swap');
    body, .gradio-container { background: linear-gradient(135deg, #f7f1e5 0%, #e8dcc7 45%, #d6c6ae 100%); color: #333333; }
    .gradio-container { font-family: 'Space Grotesk', sans-serif; }
    h1, h2, p { font-family: 'Fraunces', serif; color: #333333; }
    black_label { color: #333333; font-size: 17px; }
    """

    with gr.Blocks(css=custom_css, title="Yelp Latent Attribute Explorer") as demo:
        gr.Markdown(
            """
            # Yelp Latent Attribute Explorer
            Search businesses by abstract preference, inspect sentiment-aware attributes, and view a local knowledge-graph neighborhood.
            """
        )
        with gr.Row():
            query_box = gr.Textbox(label="Search intent", value="quiet places to work with coffee and wifi")
            top_k = gr.Slider(label="Top K", minimum=3, maximum=20, value=8, step=1)
        run_button = gr.Button("Search", variant="primary")
        results = gr.Dataframe(label="Semantic matches", interactive=False, elem_id="black_label")
        selection = gr.Dropdown(label="Choose a business to inspect", choices=[])
        graph_plot = gr.Plot(label="Knowledge graph view")
        details = gr.Markdown(label="Business details")

        run_button.click(
            run_query,
            inputs=[query_box, top_k],
            outputs=[results, selection, graph_plot, details],
        )
        selection.change(
            lambda selected: (render_selected(selected), render_business_details(selected)),
            inputs=[selection],
            outputs=[graph_plot, details],
        )

    return demo


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the Yelp latent attribute explorer app.")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = launch_app(args.artifacts_dir)
    demo.launch(server_name=args.host, server_port=args.port)
