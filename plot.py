"""
Plot results and save as HTML file
"""

import textwrap

import numpy as np
from sklearn.decomposition import TruncatedSVD
import plotly.express as px

from util import i18n


def add_score_size(plot_df, col="score", out_col="score_size", q=(0.05, 0.95)):
    s = plot_df[col].astype(float)

    lo, hi = s.quantile(q[0]), s.quantile(q[1])
    s_clip = s.clip(lo, hi)

    # normalize to [0, 1]
    s_norm = (s_clip - lo) / (hi - lo + 1e-12)

    plot_df[out_col] = s_norm
    return plot_df


def add_score_weighted_jitter(
    df,
    score_col="score",
    x_col="x",
    y_col="y",
    base_scale=0.05,
    random_state=0,
):
    """
    Apply weighted jitter based on the "score_col" to prevent points from obscuring each other.
    """
    rng = np.random.default_rng(random_state)

    scores = df[score_col].astype(float)
    s_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)

    dx = df[x_col].max() - df[x_col].min()
    dy = df[y_col].max() - df[y_col].min()

    df = df.copy()
    # high score -> smaller jitter
    jitter_scale = base_scale * (1 - s_norm)

    df[x_col] += rng.normal(0, jitter_scale * dx)
    df[y_col] += rng.normal(0, jitter_scale * dy)

    return df


def wrap_text(text: str, width: int = 60) -> str:
    """
    Insert <br> line breaks into long text for Plotly hover.
    """
    if not isinstance(text, str):
        return ""
    return "<br>".join(textwrap.wrap(text, width=width))


def plot_clusters_interactive(df, X, *, title_col="title", hover_cols=None):
    """
    Interactive 2D visualization of TF-IDF clusters using TruncatedSVD (PCA-like for sparse matrices).
    Click a point to open the paper in a new tab (url preferred, doi fallback).
    """
    if hover_cols is None:
        hover_cols = ["doi", "container_title", "score"]

    def _resolve_link(row) -> str:
        # 1) prefer 'url' in df
        url = str(row.get("url", "") or "").strip()
        if url:
            return url
        # 2) fallback to doi
        doi = str(row.get("doi", "") or "").strip()
        if not doi:
            return ""
        return doi if doi.startswith("http") else f"https://doi.org/{doi}"

    # TruncatedSVD works directly on sparse TF-IDF matrices
    svd = TruncatedSVD(n_components=2, random_state=0)
    coords = svd.fit_transform(X)

    # copy df
    plot_df = df.copy()

    # plot hover info
    plot_df["title_wrapped"] = plot_df["title"].apply(lambda t: wrap_text(t, width=60))
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]

    # add weighted jitter based on score to prevent points from obscuring each other
    plot_df = add_score_weighted_jitter(plot_df, base_scale=0.05)

    # plot cluster
    plot_df["cluster"] = plot_df["cluster"].astype(str)
    plot_df = add_score_size(plot_df, col="score", out_col="score_size")
    plot_df = plot_df.sort_values("score", ascending=False).reset_index(drop=True)

    # show paper score global rank
    plot_df["rank"] = plot_df.index + 1

    # Link fields for click + hover
    plot_df["paper_link"] = plot_df.apply(_resolve_link, axis=1)
    plot_df["link_label"] = plot_df["paper_link"].apply(
        lambda s: ("click to open" if str(s).strip() else "no link found")
    )

    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="cluster_legend",
        size="score_size",
        size_max=28,
        custom_data=[
            "cluster_legend",  # [0]
            "rank",  # [1]
            "title_wrapped",  # [2]
            "doi",  # [3]
            "container_title",  # [4]
            "score",  # [5]
            "paper_link",  # [6]
            "link_label",  # [7]
        ],
        title="Paper clusters (TF-IDF on titles)",
    )

    fig.update_traces(
        marker=dict(opacity=0.75),
        selector=dict(mode="markers"),
        hovertemplate=(
            "<b>%{customdata[2]}</b><br>"
            "Rank: %{customdata[1]}<br>"
            "Cluster: %{customdata[0]}<br>"
            "Journal: %{customdata[4]}<br>"
            "DOI: %{customdata[3]}<br>"
            "Score: %{customdata[5]:.3f}<br>"
            "Link: %{customdata[7]}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        legend=dict(
            orientation="h",  # horizontal
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        hovermode="closest",
        margin=dict(b=120),
        legend_title_text="Cluster",
        height=650,
    )

    return fig


def plot_publisher_interactive(counts, publisher_order, top_n=8):
    """
    Interactive 2D visualization of publishers and article types.
    """
    fig = px.bar(
        counts,
        x="publisher_grouped",
        y="count",
        color="article_type",
        orientation="v",
        category_orders={"publisher_grouped": list(publisher_order)},
        title="Article types by publisher",
        text="count",
    )

    fig.update_layout(
        barmode="stack",
        height=500 + 30 * top_n,
        xaxis_title="Publisher",
        yaxis_title="Number of papers",
        legend_title_text="Article type",
        legend=dict(
            x=0.85,
            y=0.99,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        ),
    )

    # Show numbers in the bar
    fig.update_traces(
        textposition="inside",
        insidetextanchor="middle",
    )

    return fig
