"""
Analyse papers collected from Crossref
"""

from datetime import date, timedelta

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.cluster import KMeans
import pandas as pd

from get_data import get_data_by_orcid
from stop_words import DOMAIN_STOP_WORDS
from clustering import (
    vectorize_titles_tfidf,
    choose_k_with_silhouette,
    top_terms_by_cluster,
)
from plot import plot_clusters_interactive, plot_publisher_interactive
from util import map_article_type, load_followed_authors, add_followed_author_flags


def cluster_analysis(df: pd.DataFrame, path: str = "./html/clusters.html"):
    """
    Performs clustering on a dataframe and saves an interactive plot as html.

    Returns:
        df_out: copy of df with columns ['cluster', 'cluster_legend']
        cluster_terms: dict[int, list[str]]
        k: chosen number of clusters
    """
    print(f"\nClustering on {len(df)} papers")
    df_out = df.copy()

    stop_words = list(ENGLISH_STOP_WORDS.union(DOMAIN_STOP_WORDS))
    X, vectorizer = vectorize_titles_tfidf(
        df_out, ngram_range=(1, 2), stop_words=stop_words
    )

    k, silhouette_scores = choose_k_with_silhouette(
        X, k_candidates=range(3, 8), delta=0.02
    )
    print(f"\tBest k found (based on silhouette scores): {k}")

    kmeans = KMeans(n_clusters=k, random_state=42)
    df_out["cluster"] = kmeans.fit_predict(X)
    print(f"\tK-Means clustering done!")

    cluster_terms = top_terms_by_cluster(df_out, X, vectorizer, top_n=4)

    df_out["cluster_legend"] = df_out["cluster"].map(
        lambda c: f"Cluster {c}: " + ", ".join(cluster_terms[c])
    )

    fig = plot_clusters_interactive(df_out, X)
    # Add JavaScript code to open the URL in the plot when clicked.
    post_script = """
    var plot = document.getElementsByClassName('plotly-graph-div')[0];
    if (plot) {
      plot.on('plotly_click', function(e) {
        try {
          var url = e.points[0].customdata[6]; // paper_link
          if (url && String(url).trim().length > 0) {
            window.open(url, '_blank');
          }
        } catch (err) {
          // do nothing
        }
      });
    }
    """
    fig.write_html(
        path,
        include_plotlyjs="cdn",
        full_html=True,
        post_script=post_script,
    )
    print(f"\tSaved clusters interactive html file: {path}")

    return df_out, cluster_terms, k


def top_picks_by_cluster(
    df: pd.DataFrame,
    sort_by: str,
    top_n: int = 3,
    ascending: bool = False,
) -> dict[int, pd.DataFrame]:
    """
    Group by cluster and take top_n rows per cluster after sorting.

    Returns:
        dict: {cluster_id: df_top_cluster}
    """
    if "cluster" not in df.columns:
        raise ValueError(
            "df must contain a 'cluster' column. Run cluster_analysis first."
        )

    if sort_by not in df.columns:
        raise ValueError(f"sort_by column '{sort_by}' not found in df.")

    out = {}
    for c, g in df.groupby("cluster", sort=False):
        g2 = g.sort_values(sort_by, ascending=ascending, na_position="last").head(top_n)
        out[int(c)] = g2
    return out


def publisher_analysis(
    df: pd.DataFrame, path: str = "./html/publishers.html", top_n: int = 8
):
    """
    Performs publisher analysis on a dataframe and
    saves the results as a html file.
    """
    print(f"\nPublisher analysis on {len(df)} papers")
    df_plot = df.copy()
    df_plot["article_type"] = df_plot["type"].apply(map_article_type)

    top_publishers = df_plot["publisher"].value_counts().head(top_n).index

    df_plot = df_plot.copy()
    df_plot["publisher_grouped"] = df_plot["publisher"].where(
        df_plot["publisher"].isin(top_publishers), "Other publishers"
    )

    counts = (
        df_plot.groupby(["publisher_grouped", "article_type"])
        .size()
        .reset_index(name="count")
    )

    publisher_order = (
        counts.groupby("publisher_grouped")["count"]
        .sum()
        .sort_values(ascending=False)
        .index
    )

    # Put Other at the end
    publisher_order = [p for p in publisher_order if p != "Other publishers"] + [
        "Other publishers"
    ]

    fig = plot_publisher_interactive(counts, publisher_order, top_n)
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"\tSaved publisher interactive html file: {path}")


def followed_authors_analysis(
    df: pd.DataFrame,
    *,
    followed_authors_path: str = "./config/followed_authors.yaml",
    from_date: str | None = None,
    types: list[str] | None = None,
    rows_per_author: int = 50,
    mailto: str | None = None,
):
    """
    Filter 'Papers by followed authors' from the existing dataframe.

    Additional: Query authors' new journal/conference papers from the past week based on ORCIDs in followed_authors.yaml.

    Notes
    -----
    ORCID queries only cover works where the publisher has submitted the ORCID to Crossref metadata

    Returns
    -------
    df_followed_in_results, df_followed_recent_by_orcid
    """
    print(f"\nFollowed authors analysis on {len(df)} papers")
    if from_date is None:
        from_date = (date.today() - timedelta(days=7)).isoformat()

    if types is None:
        types = ["journal-article", "proceedings-article"]

    # Load followed authors
    followed_authors = load_followed_authors(followed_authors_path)

    # Find followed authors in the current search results dataframe
    df_plot = add_followed_author_flags(df.copy(), followed_authors)
    df_followed_in_results = (
        df_plot[df_plot["is_followed_author_paper"]]
        .sort_values("score", ascending=False)
        .copy()
    )

    # Use ORCID to additionally retrieve "New works added by the author in the past week"
    rows = []
    for a in followed_authors:
        orcid = (a.get("orcid") or "").strip()
        name = (a.get("name") or "").strip()
        if not orcid:
            continue

        try:
            df_a = get_data_by_orcid(
                orcid=orcid,
                types=types,
                from_date=from_date,
                rows=rows_per_author,
                mailto=mailto,
                sort="created",
                order="desc",
            )
        except Exception as e:
            print(f"\t[followed-authors] ORCID query failed for {name or orcid}: {e}")
            continue

        if df_a.empty:
            continue

        df_a["followed_author_orcid"] = orcid
        df_a["followed_author_name"] = name or orcid
        rows.append(df_a)

    if rows:
        df_followed_recent_by_orcid = pd.concat(rows, ignore_index=True)
        # Deduplication (where the same article may be hit by
        # multiple authors or overlap with the main search)
        if "doi" in df_followed_recent_by_orcid.columns:
            df_followed_recent_by_orcid["doi_norm"] = (
                df_followed_recent_by_orcid["doi"].astype(str).str.lower()
            )
            df_followed_recent_by_orcid = df_followed_recent_by_orcid.drop_duplicates(
                "doi_norm"
            )
    else:
        df_followed_recent_by_orcid = pd.DataFrame()

    print("\tPapers by followed authors (found in keyword results)")
    if df_followed_in_results.empty:
        print("\t\t- None")
    else:
        for _, row in df_followed_in_results.iterrows():
            print(
                f"\t\t- {row['title']} ({row['container_title']}) {row['doi']} [{row.get('followed_author_label','')}]"
            )

    print("\tPapers by followed authors (found with ORCID)")
    if df_followed_recent_by_orcid.empty:
        print("\t\t- None")
    else:
        for _, row in df_followed_recent_by_orcid.sort_values(
            "created_date", ascending=False
        ).iterrows():
            print(
                f"\t\t- {row['title']} ({row['container_title']}) {row['doi']}  — {row.get('followed_author_name','')}"
            )

    return df_followed_in_results, df_followed_recent_by_orcid
